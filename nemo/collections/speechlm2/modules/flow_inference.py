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
import os
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


class AudioDecoder(torch.nn.Module): # from token to wav
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, block_size=10, device="cuda", causal_conv=False, learnable_prompt=False, config_overrides=None):
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
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device), strict=not (learnable_prompt))
        self.hift = self.scratch_configs['hift']
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
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

    def token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192), finalize=False):

        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device).repeat(token.shape[0]),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device).repeat(token.shape[0]),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device).repeat(token.shape[0]),
                                      embedding=embedding.to(self.device))

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)

        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            # if self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            tts_speech = tts_speech[:, :-self.source_cache_len]

        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
            # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel

    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def stream_inference(self, token, this_uuid, flow_prompt_speech_token, prompt_speech_feat, spk_emb, pad_args=None):


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

            # print(tts_token.size())

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=flow_prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device),
                                                 embedding=spk_emb,
                                                 finalize=is_finalize)

            prev_mel = tts_mel
            prev_speech = tts_speech
            prev_idx = idx + block_size
            # print(tts_mel.size())

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1)

        return tts_speech
