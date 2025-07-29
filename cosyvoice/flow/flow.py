# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask


class MaskedDiffWithXvec(torch.nn.Module):
    """
    Main model for text-to-speech (TTS) using a Masked Diffusion Flow with speaker embedding (xvec).
    This model converts input text tokens into mel-spectrograms, supporting speaker adaptation via xvec.

    Architecture Overview:
    - Encoder: Converts input text tokens into hidden representations.
    - Length Regulator: Adjusts encoder output to match target mel-spectrogram length.
    - Decoder: Conditional flow-based decoder generates mel-spectrograms from regulated hidden states, speaker embedding, and optional conditions.

    Usage:
    - Training: Call forward() with a batch dict to compute loss.
    - Inference: Call inference() to synthesize mel-spectrograms from text tokens and speaker embedding.

    Args:
        input_size: Dimension of input token embeddings.
        output_size: Dimension of output mel-spectrogram features.
        spk_embed_dim: Dimension of input speaker embedding (xvec).
        output_type: Output feature type, e.g., "mel".
        vocab_size: Vocabulary size for input tokens.
        input_frame_rate: Frame rate for input tokens.
        only_mask_loss: Whether to use only masked loss.
        encoder: Encoder module (e.g., Transformer/Conformer).
        length_regulator: Module to align encoder output to target length.
        decoder: Conditional flow-based decoder module.
        decoder_conf: Decoder configuration dictionary.
        mel_feat_conf: Mel-spectrogram feature configuration.
    """
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1, 'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}), 'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64, 'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050, 'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")

        # Embedding layer: maps token ids to embeddings
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        # Speaker embedding projection: projects xvec to output_size
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        # Encoder: encodes input token embeddings
        self.encoder = encoder
        # Project encoder output to output_size (for decoder input)
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        # Decoder: conditional flow-based decoder (e.g., UNet, CFM)
        self.decoder = decoder
        # Length regulator: aligns encoder output to target mel length
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Training forward pass.
        Args:
            batch: Dictionary containing input tokens, lengths, features, and speaker embedding.
            device: Target device.
        Returns:
            Dict with 'loss' key.
        """
        # 1. Prepare input tokens and features
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device).transpose(1, 2)  # [B, T, mel] -> [B, mel, T]
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 2. Speaker embedding normalization and projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 3. Token embedding with padding mask
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # 4. Encoder: text tokens -> hidden states
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        # 5. Length regulator: match hidden states to target mel length
        h, h_lengths = self.length_regulator(h, feat_len)

        # 6. Prepare mask for decoder and interpolate features to match length
        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)

        # 7. Prepare conditional input (prompted mel frames)
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.8 * j))
            conds[i, :index] = feat[i, :index]
            # If causal_mask is enabled, mask out future frames for causality
            if getattr(self, 'causal_mask', False):
                mask[i, index:] = 0
        conds = conds.transpose(1, 2)  # [B, mel, T] -> [B, T, mel]

        # 8. Decoder: compute loss between predicted and target mel-spectrogram
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),  # [B, T, mel]
            mask.unsqueeze(1),                  # [B, 1, T]
            h.transpose(1, 2).contiguous(),     # [B, T, hidden]
            embedding,                          # [B, output_size]
            cond=conds                          # [B, T, mel]
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding):
        """
        Inference (synthesis) pass.
        Args:
            token: Target text tokens [B, T]
            token_len: Lengths of target tokens [B]
            prompt_token: Prompt tokens (for context) [B, T_p]
            prompt_token_len: Lengths of prompt tokens [B]
            prompt_feat: Prompt mel-spectrogram frames [B, T_p, mel]
            prompt_feat_len: Lengths of prompt features [B]
            embedding: Speaker embedding (xvec) [B, spk_embed_dim]
        Returns:
            Synthesized mel-spectrogram [B, mel, T]
        """
        # 1. Speaker embedding normalization and projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 2. Concatenate prompt and target tokens
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # 3. Encoder: text tokens -> hidden states
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        # 4. Estimate output mel length from token length
        feat_len = (token_len / self.input_frame_rate * 22050 / 256).int()
        h, h_lengths = self.length_regulator(h, feat_len)

        # 5. Prepare conditional input (prompted mel frames)
        conds = torch.zeros([token.shape[0], feat_len.max().item(), self.output_size], device=token.device)
        if prompt_feat.shape[1] != 0:
            for i, j in enumerate(prompt_feat_len):
                conds[i, :j] = prompt_feat[i]
        conds = conds.transpose(1, 2)  # [B, T, mel] -> [B, mel, T]

        mask = (~make_pad_mask(feat_len)).to(h)

        # 6. Decoder: generate mel-spectrogram
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),  # [B, hidden, T]
            mask=mask.unsqueeze(1),             # [B, 1, T]
            spks=embedding,                     # [B, output_size]
            cond=conds,                         # [B, mel, T]
            n_timesteps=10                      # Number of reverse diffusion steps
        )
        # Remove prompt frames from output if prompt was used
        if prompt_feat.shape[1] != 0:
            feat = feat[:, :, prompt_feat.shape[1]:]
        return feat
