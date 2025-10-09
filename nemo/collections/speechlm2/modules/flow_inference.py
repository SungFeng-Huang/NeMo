import os
import sys
from typing import Dict, Optional

import torch
import torchaudio
import numpy as np
import re
from hyperpyyaml import load_hyperpyyaml
import uuid
from collections import defaultdict
from .causal_conv import CausalConfigManager, CausalConvConverter
import yaml
import tempfile

import json
import hydra
from omegaconf import OmegaConf

def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class StreamStateManager:
    """
    Simplified unified state manager for streaming inference
    """
    
    def __init__(self, mel_overlap_len, mel_cache_len, source_cache_len):
        self.mel_overlap_len = mel_overlap_len
        self.mel_cache_len = mel_cache_len
        self.source_cache_len = source_cache_len
        
        # Simple state storage
        self.mel_overlap_dict = {}
        self.hift_cache_dict = {}
    
    def get_mel_overlap(self, uuid_str):
        """Get stored mel-spectrogram overlap"""
        return self.mel_overlap_dict.get(uuid_str)
    
    def set_mel_overlap(self, uuid_str, mel_spectrogram):
        """Store mel-spectrogram overlap"""
        self.mel_overlap_dict[uuid_str] = mel_spectrogram[:, :, -self.mel_overlap_len:]
    
    def get_hift_cache(self, uuid_str):
        """Get stored HiFi-GAN cache"""
        return self.hift_cache_dict.get(uuid_str)
    
    def set_hift_cache(self, uuid_str, mel, source, speech):
        """Store HiFi-GAN cache"""
        self.hift_cache_dict[uuid_str] = {
            'mel': mel[:, :, -self.mel_cache_len:],
            'source': source[:, :, -self.source_cache_len:],
            'speech': speech[:, -self.source_cache_len:]
        }
    
    def clear_state(self, uuid_str):
        """Clear all state for the given UUID"""
        self.mel_overlap_dict.pop(uuid_str, None)
        self.hift_cache_dict.pop(uuid_str, None)


class AudioDecoder(torch.nn.Module): # from token to wav
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, device="cuda", block_size=10, causal_conv=False, learnable_prompt=False, config_overrides=None):
        super().__init__()
        self.device = device

        # Initialize causal configuration manager
        causal_config = {
            'causal_mode': True,
            'affected_modules': ['decoder'],
            'conversion_strategy': 'converter'
        }
        if config_overrides is None:
            config_overrides = {}
        if causal_conv:
            config_overrides['flow.encoder.causal'] = True    # causal conv
        if learnable_prompt:
            config_overrides['flow.learnable_prompt'] = True    # learnable prompt

        try:
            with open(config_path, 'r') as f:
                scratch_configs = load_hyperpyyaml(f)
            scratch_configs['flow'].load_state_dict(torch.load(flow_ckpt_path, map_location=self.device), strict=True)
        except Exception as e:
            print(f"Error loading flow model: {e}")
            raise e

        # Load and potentially modify config before instantiation
        # Check if config_overrides is not None and not an empty dict
        if config_overrides is not None and len(config_overrides) > 0:
            del scratch_configs
            # Release CUDA memory before reloading configs/models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            self.scratch_configs = self._load_config_with_overrides(config_path, config_overrides)
        else:
            self.scratch_configs = scratch_configs

        # Load models
        self.flow = self.scratch_configs['flow']
        if flow_ckpt_path and os.path.isfile(flow_ckpt_path):
            self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device), strict=not (learnable_prompt))
        self.hift = self.scratch_configs['hift']
        if hift_ckpt_path and os.path.isfile(hift_ckpt_path):
            self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        
        # Initialize unified stream state manager
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.block_size = block_size
        
        # Initialize unified state manager
        self.state_manager = StreamStateManager(
            mel_overlap_len=self.mel_overlap_len,
            mel_cache_len=self.mel_cache_len,
            source_cache_len=self.source_cache_len
        )

        if causal_conv:
            self.causal_config_manager = CausalConfigManager()
            self.causal_config_manager.update_config(causal_config)
            self.causal_config_manager.apply_to_model(self.flow)

    def _load_config_with_overrides(self, config_path, overrides):
        """
        Load config with overrides before instantiation.
        This function directly parses the YAML file using standard I/O,
        applies the overrides, and returns the modified config object.
        
        Args:
            config_path: Path to the original YAML config file
            overrides: Dict of parameter overrides (e.g., {'flow.encoder.causal': True})        
        Returns:
            Modified config object with overrides applied
        """        
        # Convert overrides to Hydra format
        override_list = []
        for key_path, value in overrides.items():
            override_list.append(f"{key_path}={value}")
        
        # Read the original config file as string to preserve hyperpyyaml syntax
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Apply overrides by modifying the YAML content string
        for key_path, value in overrides.items():
            # Convert value to YAML string representation
            if isinstance(value, bool):
                yaml_value = str(value).lower()
            elif isinstance(value, str):
                yaml_value = f"'{value}'"
            elif isinstance(value, (int, float)):
                yaml_value = str(value)
            elif isinstance(value, (list, dict)):
                yaml_value = yaml.dump(value, default_flow_style=True).strip()
            else:
                yaml_value = str(value)
            
            # Find and replace the specific key in the YAML content
            config_content = self._replace_yaml_value(config_content, key_path, yaml_value)
        
        # Create temporary file with modified config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write(config_content)
            temp_config_path = temp_file.name
        
        try:
            # Load modified config with hyperpyyaml
            with open(temp_config_path, 'r') as f:
                config = load_hyperpyyaml(f)
            return config
        finally:
            # Clean up temporary file
            os.unlink(temp_config_path)

    def _replace_yaml_value(self, yaml_content, key_path, new_value):
        """
        Replace a specific key's value in YAML content while preserving structure
        
        Args:
            yaml_content: The original YAML content as string
            key_path: Dot-separated path to the key (e.g., 'flow.encoder.causal')
            new_value: The new value to set
            
        Returns:
            Modified YAML content as string
        """
        keys = key_path.split('.')
        lines = yaml_content.split('\n')
        modified_lines = []
        
        # Track current nesting level and key path
        current_level = 0
        current_key_path = []
        found_target = False
        found_target_key_parent_path = False
        
        for i, line in enumerate(lines):
            # Count indentation (assuming 4 spaces per level)
            indent = len(line) - len(line.lstrip())
            level = indent // 4
            
            # Update current key path based on indentation level
            while len(current_key_path) > level:
                current_key_path.pop()
            
            # Check if this line contains a key
            if ':' in line and line.strip():
                line_key = line.strip().split(':')[0]
                if len(current_key_path) == level:
                    current_key_path.append(line_key)
                elif len(current_key_path) > level:
                    current_key_path[level] = line_key
            
            # Check if we're at the target level and this line contains our key
            # if level == len(keys) - 1 and keys[-1] in line and ':' in line:
            if not found_target:
                # Check if the current key parent path matches the target key parent path up to the current level
                if self._is_correct_key_path(current_key_path[:len(keys) - 1], keys[:-1]):
                    found_target_key_parent_path = True
                    # Verify this is the correct key by checking the full path
                    if self._is_correct_key_path(current_key_path, keys):
                        # Replace the value after the colon
                        colon_pos = line.find(':')
                        new_line = f"{line[:colon_pos+1]} {new_value}"
                        modified_lines.append(new_line)
                        found_target = True
                        continue
                elif found_target_key_parent_path:
                    # If we found the parent key path, but not the target key before exiting the loop,
                    # it means the target key is not in the config file.
                    # If we didn't find the key, append it at the appropriate level
                    # Build the full path structure
                    target_level = len(keys) - 1
                    indent_str = '    ' * target_level  # 4 spaces per level
                    new_line = f"{indent_str}{keys[-1]}: {new_value}"
                    modified_lines.append(new_line)
                    found_target = True
            
            modified_lines.append(line)
        
        
        return '\n'.join(modified_lines)
    
    def _is_correct_key_path(self, current_key_path, target_keys):
        """
        Check if the current key path matches the target key path
        
        Args:
            current_key_path: List of keys in the current path (e.g., ['flow', 'encoder'])
            target_keys: List of target keys (e.g., ['flow', 'encoder', 'causal'])
            
        Returns:
            True if the paths match, False otherwise
        """
        # Check if the current path matches the target path up to the current level
        if len(current_key_path) != len(target_keys):
            return False
        
        # Check if all keys in the current path match the target path
        for i, key in enumerate(current_key_path):
            if key != target_keys[i]:
                return False
        
        return True

    @torch.inference_mode()
    def token2wav(
        self,
        token: torch.Tensor,
        uuid: str,
        prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_feat: torch.Tensor = torch.zeros(1, 0, 80),
        embedding: torch.Tensor = torch.zeros(1, 192),
        finalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode tokens to waveform using CosyVoice2 flow + HiFT vocoder.

        Args:
            token: [B, T_tok] discrete tokens
            uuid: session id for stream cache (unused for finalize=True)
            prompt_token: [B, Tp]
            prompt_feat: [B, Tp_mel, 80] or [B, Tp_mel, 80] (expects [B, Tp_mel, 80])
            embedding: [B, 192] speaker embedding
            finalize: end of stream flag; if True, process all remaining frames

        Returns:
            tts_speech_output: [B, T_wav] at 22050 Hz
            tts_mel_output: [B, 80, T_mel] generated mel
        """
        # Step 1: Generate mel-spectrogram from tokens
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device).repeat(token.shape[0]),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device).repeat(token.shape[0]),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device).repeat(token.shape[0]),
                                      embedding=embedding.to(self.device))

        # Step 2: Apply mel-spectrogram overlap for smooth transitions
        mel_overlap = self.state_manager.get_mel_overlap(uuid)
        if mel_overlap is not None:
            tts_mel = fade_in_out(tts_mel, mel_overlap, self.mel_window)
        
        # Step 3: Apply HiFi-GAN cache for continuous audio generation
        hift_cache = self.state_manager.get_hift_cache(uuid)
        if hift_cache is not None:
            hift_cache_mel, hift_cache_source = hift_cache['mel'], hift_cache['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)

        # Step 4: Process based on whether this is the final chunk
        if not finalize:
            # Store overlap for next iteration
            self.state_manager.set_mel_overlap(uuid, tts_mel)
            
            # Remove overlap from current output
            tts_mel_output = tts_mel[:, :, :-self.mel_overlap_len]
            
            # Generate audio using HiFi-GAN
            tts_speech, tts_source = self.hift.inference(mel=tts_mel_output, cache_source=hift_cache_source)
            
            # Update HiFi-GAN cache
            self.state_manager.set_hift_cache(uuid, tts_mel_output, tts_source, tts_speech)
            
            # Remove cache from output
            tts_speech_output = tts_speech[:, :-self.source_cache_len]
        else:
            # Final chunk - generate complete audio
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            tts_speech_output = tts_speech
            tts_mel_output = tts_mel
            
            # Clear all state for this UUID
            self.state_manager.clear_state(uuid)

        return tts_speech_output, tts_mel_output

    @torch.inference_mode()
    def stream_inference(
        self,
        token: torch.Tensor,
        this_uuid: str,
        prompt_speech_token: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        spk_emb: torch.Tensor,
        pad_args: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Streaming decode tokens to waveform using flow + HiFT with overlap-fade.

        Args:
            token: [B, T_tok]
            this_uuid: stream session id
            prompt_speech_token: [B, Tp]
            prompt_speech_feat: [B, Tp_mel, 80]
            spk_emb: [B, 192]
            pad_args: padding arguments
        Returns:
            wav22050: [B, T_wav] at 22050 Hz
        """


        tts_speechs = []
        tts_mels = []

        # block_size = self.flow.encoder.block_size
        block_size = self.block_size
        prev_mel = None

        prev_idx = 0
        start_idx = 0 if pad_args is None else self.flow.encoder.block_size

        for idx in range(start_idx, token.size(1), block_size):
            # current block: idx ~ idx + block_size
            # if padding: prev_idx = enc_block_size, 
            # first block: 0 ~ enc_block_size + block_size
            #   --> next prev_idx = enc_block_size + block_size
            # other blocks: prev_idx ~ idx + block_size
            tts_token = token[:, prev_idx:idx + block_size]
            
            # Determine if this is the final block
            is_finalize = (idx + block_size >= token.size(-1))

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                prompt_speech_token = token[:, :idx]

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device),
                                                 embedding=spk_emb,
                                                 finalize=is_finalize)

            prev_mel = tts_mel
            # prev_speech = tts_speech
            prev_idx = idx + block_size
            # print(tts_mel.size())

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1)

        return tts_speech


    @torch.inference_mode()
    def offline_inference(
        self,
        token: torch.Tensor,
    ) -> torch.Tensor:
        """Non-streaming inference: run CosyVoice2 flow in finalize=True once and vocoder once.
        Returns wav at 22050 Hz for direct comparison with streaming path.
        """
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()