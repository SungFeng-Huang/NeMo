import json
import os
import tempfile
import types
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import copy

import torch
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import torchaudio

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging

# English text normalizer
from nemo.collections.asr.modules.rnnt import RNNTDecoder
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_nemo_attention_dropout_hooks(model):
    """Add hooks to capture attention weights from Conformer layers"""
    
    # Function to capture attention weights
    def capture_attention(self, module, input, output):
        # For nemo multi-head attention, the attention weights are not stored, so extract from the attention dropout's input
        # The exact structure depends on the implementation
        assert isinstance(input, tuple) and len(input) > 0
        self.attention_weights = input[0].detach().cpu().numpy()
        
    # Add hooks to all self-attention modules in the encoder
    for i, layer in enumerate(model.encoder.layers):
        if hasattr(layer, 'self_attn'):
            # Register forward hook to capture attention
            layer.self_attn.attention_weights = None
            layer.self_attn.dropout.register_forward_hook(
                types.MethodType(capture_attention, layer.self_attn)
            )
    
    return model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class StarRNNT:
    """Implementation of STAR for RNN-T model"""
    def __init__(
        self, 
        asr_model: EncDecRNNTModel,
        tokenizer: TokenizerSpec,
        threshold: float = 2.0,
        tau: float = 10.0,
    ):
        self.asr_model = asr_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.tau = tau
        self.state_dict = copy.deepcopy(asr_model.state_dict())
        
    def get_star_scores(self, hyp: Hypothesis, avg_attn=None, mode='decoder', causal=True) -> List[float]:
        """Calculate STAR scores from a hypothesis
        
        Args:
            hyp: Hypothesis object with token confidences
            encoder_outputs: Optional outputs from encoder with attention weights
            
        Returns:
            List of STAR scores for each token
        """
        # Get confidence scores (probs)
        probs = self.get_confidence_score(hyp, mode)
        
        # Get attention weights
        weights = self.get_attentive_score(hyp, avg_attn, probs, mode, causal)
        
        # Calculate STAR scores (final_weights)
        final_weights = []
        for ci, ai in zip(probs, weights):
            c_over_a, a_over_c = ci * ci / ai if ai > 0 else 0, ai * ai / ci if ci > 0 else 0
            conflict = (sigmoid((c_over_a - self.threshold) * self.tau) + sigmoid((a_over_c - self.threshold) * self.tau)) * ai
            no_conflict = (sigmoid((self.threshold - c_over_a) * self.tau) * sigmoid((self.threshold - a_over_c) * self.tau)) * ai * np.exp((ci - ai) / self.tau)
            final_weights.append(conflict + no_conflict)
        
        return final_weights

    def get_attentive_score(self, hyp, avg_attn, probs, mode, causal):
        weights = []
        
        # Extract relevant attention scores for each token
        # This mapping can be complex and depends on how the model aligns
        # encoder frame positions to decoded tokens
        if mode == 'encoder':
            # avg_attn = avg_attn[timestep, timestep]
            weights = self.get_encoder_attentive_score(hyp, avg_attn, causal=causal)
        
        elif mode == 'decoder':
            weights = self.get_decoder_attentive_score(hyp, avg_attn, causal=causal)

            # If direct attention extraction failed or returned empty, fall back to timestamp-based approach
            if not weights:
                # If timestamps available, use them to estimate attention weights
                weights = self.get_decoder_timestamp_score(hyp)
        
        # If still no weights, use uniform weights
        if not weights:
            weights = [1.0] * len(probs)
        
        # Normalize weights
        if weights:
            mean_weights = sum(weights) / len(weights)
            weights = [round(w / mean_weights, 3) for w in weights]
        return weights

    def get_decoder_timestamp_score(self, hyp):
        weights = []

        if hasattr(hyp, 'timestamp') and hyp.timestamp:
            timestamp = hyp.timestamp
            if isinstance(timestamp, dict):
                timestamp = timestamp.get('timestep', [])
                    
            if timestamp:
                # Calculate normalized time progression as attention estimate
                max_time = max(timestamp) if timestamp else 1
                weights = [round(t / max_time, 3) for t in timestamp]
        return weights

    def get_decoder_attentive_score(self, hyp, avg_attn, causal=True):
        weights = []
        timestep = hyp.timestep['timestep']
        if hasattr(hyp, 'alignments') and hyp.alignments:
            for align_idx in timestep:
                # For RNNT, alignments can be 2D
                # We need to find corresponding encoder frames
                # avg_attn[i, :] == 1
                if causal:
                    weight = avg_attn[align_idx, :align_idx].sum() + avg_attn[align_idx+1:, align_idx].sum()

                elif 0 <= align_idx < avg_attn.shape[-1]:
                    # For simpler cases, bi-directional mapping
                    # Sum attention for this position
                    weight = avg_attn[:, align_idx].sum().item()

                weights.append(weight)
        return weights

    def get_encoder_attentive_score(self, hyp, avg_attn, causal=True):
        weights = []
        timestep = hyp.timestep['timestep']
        # avg_attn[i, :] == 1
        if hasattr(hyp, 'alignments') and hyp.alignments:
            for i, align_idx in enumerate(hyp.alignments):
                if causal:
                    # For RNNT, alignments can be 2D
                    # We need to find corresponding encoder frames
                    weight = avg_attn[i, :i].sum() + avg_attn[i+1:, i].sum()
                elif 0 <= i < avg_attn.shape[-1]:
                    # For simpler cases, direct mapping
                    # Sum attention for this position
                    weight = avg_attn[:, i].sum().item()
                weights.append(weight)
        return weights

    def get_confidence_score(self, hyp, mode):
        if mode == 'decoder':
            probs = hyp.token_confidence if hyp.token_confidence is not None else []
        elif mode == 'encoder':
            probs = [ci[-1] for ci in hyp.frame_confidence] if hyp.frame_confidence is not None else []
        
        # If token_confidence is empty, try to use frame_confidence
        if not probs and hyp.non_blank_frame_confidence:
            probs = hyp.non_blank_frame_confidence
        
        if not probs:
            # No confidence scores available, return uniform weights
            return [1.0] * (len(hyp.y_sequence) - 1)  # -1 to exclude blank token
        
        # Normalize confidence scores
        mean_probs = sum(probs) / len(probs)
        probs = [round(p / mean_probs, 3) for p in probs]
        return probs
    
    def generate_pseudo_labels(self, audio: torch.Tensor, sample_rate: int = 16000) -> Tuple[str, List[float], float, int]:
        """Generate pseudo-labels with STAR scores for an audio sample
        
        Args:
            audio: Audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            tuple: (transcription, star_scores, avg_wer, diversity)
        """
        # Resample audio if needed
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
            
        # Put model in evaluation mode and process audio
        self.asr_model.eval()
        with torch.no_grad():
            # Set the model to output attention weights
            self.asr_model.encoder._capture_attention = True
            
            # Process audio
            features = self.asr_model.preprocessor(audio.unsqueeze(0).to(device))
            encoded, encoded_len = self.asr_model.encoder(features)
            
            # Store encoder outputs with attention
            encoder_outputs = encoded  # This should now contain attention_weights attribute
            
            # Reset attention capturing to avoid memory issues
            self.asr_model.encoder._capture_attention = False
            
            # Get beam search results with confidence scores
            beam_results = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
                calculate_token_confidence=True
            )
            
            # Get the best hypothesis
            hyp = beam_results[0][0]  # First sample, best hypothesis
            
            # Calculate STAR scores
            star_scores = self.get_star_scores(hyp, encoder_outputs)
            
            # Get transcription
            transcription = hyp.text if hyp.text else ''
            
            # Generate multiple outputs with small noise for diversity measurement
            avg_wer, generated_texts = 0, []
            for _ in range(5):
                # Add small noise to model weights
                new_state_dict = copy.deepcopy(self.state_dict)
                for k in new_state_dict.keys():
                    if torch.is_tensor(new_state_dict[k]) and new_state_dict[k].numel() > 0:
                        std = torch.std(new_state_dict[k])
                        noise = torch.randn_like(new_state_dict[k])
                        new_state_dict[k] = new_state_dict[k] + noise * std * 0.1
                
                # Load noisy weights
                self.asr_model.load_state_dict(new_state_dict)
                
                # Generate transcription with noisy model
                with torch.no_grad():
                    noisy_results = self.asr_model.transcribe(audio.unsqueeze(0).cpu().numpy())
                    noisy_text = noisy_results[0]
                    generated_texts.append(noisy_text)
                    avg_wer += calculate_wer([transcription], [noisy_text]) / 5
            
            # Restore original weights
            self.asr_model.load_state_dict(self.state_dict)
            
            # Calculate diversity as number of unique transcripts
            diversity = len(set(generated_texts))
            
        return transcription, star_scores, avg_wer, diversity


@torch.no_grad()
def star_transcribe(
    asr_model,
    paths2audio_files: List[str],
    batch_size: int = 4,
    return_hypotheses: bool = False,
    partial_hypothesis: Optional[List['Hypothesis']] = None,
    num_workers: int = 0,
    channel_selector: Optional[ChannelSelectorType] = None,
    augmentor: DictConfig = None,
    verbose: bool = True,
) -> Tuple[List[str], Optional[List['Hypothesis']]]:
    """
    Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

    Args:

        paths2audio_files: (a list) of paths to audio files. \
    Recommended length per file is between 5 and 25 seconds. \
    But it is possible to pass a few hours long file if enough GPU memory is available.
        batch_size: (int) batch size to use during inference. \
    Bigger will result in better throughput performance but would use more memory.
        return_hypotheses: (bool) Either return hypotheses or text
    With hypotheses can do some postprocessing like getting timestamp or rescoring
        num_workers: (int) number of workers for DataLoader
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
        verbose: (bool) whether to display tqdm progress bar
    Returns:
        Returns a tuple of 2 items -
        * A list of greedy transcript texts / Hypothesis
        * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
    """
    if paths2audio_files is None or len(paths2audio_files) == 0:
        return {}

    # We will store transcriptions here
    hypotheses = []
    all_hypotheses = []
    star_scores = []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    dither_value = asr_model.preprocessor.featurizer.dither
    pad_to_value = asr_model.preprocessor.featurizer.pad_to

    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    try:
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        # Switch model to evaluation mode
        asr_model.eval()
        # Freeze the encoder and decoder modules
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        asr_model.joint.freeze()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        star_rnnt = StarRNNT(asr_model, asr_model.tokenizer)
        # Work in tmp directory - will store manifest file there
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                for audio_file in paths2audio_files:
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                    fp.write(json.dumps(entry) + '\n')

            config = {
                'paths2audio_files': paths2audio_files,
                'batch_size': batch_size,
                'temp_dir': tmpdir,
                'num_workers': num_workers,
                'channel_selector': channel_selector,
            }

            if augmentor:
                config['augmentor'] = augmentor

            temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
            for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=(not verbose)):
                encoded, encoded_len = asr_model.forward(
                    input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                )
                best_hyp, all_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(
                    encoded,
                    encoded_len,
                    return_hypotheses=return_hypotheses,
                    partial_hypotheses=partial_hypothesis,
                )

                attn_weights = asr_model.encoder.layers[-1].self_attn.attention_weights

                for i, hyp in enumerate(best_hyp):
                    # Get the attention weights for the current hypothesis
                    avg_attn = attn_weights.mean(axis=1)
                    avg_attn = avg_attn[i, :encoded_len[i], :encoded_len[i]]
                    for _mode in ['encoder', 'decoder']:
                        for causal in [True, False]:
                            star_score = star_rnnt.get_star_scores(hyp, avg_attn, mode=_mode, causal=causal)
                            setattr(hyp, f"score_{_mode}_{causal}", 0 if len(star_score) == 0 else sum(star_score)/len(star_score))
                    hypotheses.append(hyp)

                if all_hyp is not None:
                    all_hypotheses += all_hyp
                else:
                    all_hypotheses += best_hyp

                del encoded
                del test_batch
    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        asr_model.preprocessor.featurizer.dither = dither_value
        asr_model.preprocessor.featurizer.pad_to = pad_to_value

        logging.set_verbosity(logging_level)
        if mode is True:
            asr_model.encoder.unfreeze()
            asr_model.decoder.unfreeze()
            asr_model.joint.unfreeze()

    return hypotheses, all_hypotheses