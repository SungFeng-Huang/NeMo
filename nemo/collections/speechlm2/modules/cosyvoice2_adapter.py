import os
import sys
import random
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from collections import defaultdict


# NeMo resampler to keep existing validation pipeline (expects 22050 Hz)
from nemo.collections.audio.parts.utils.resampling import resample as nemo_resample
from nemo.utils import logging
from nemo.utils.formatters.base import BaseFormatter, BaseNeMoFormatter, DebugNeMoFormatter

class BaseNeMoFormatterWithRelativePath(BaseFormatter):
    DEFAULT_FORMAT = "%(color)s[NeMo %(levelname)1.1s %(asctime)s %(pathname)s:%(lineno)d][%(funcName)s()]%(end_color)s %(message)s"
    def format(self, record):
        # Find NeMo root directory by looking for nemo package
        nemo_root = None
        current_path = os.path.dirname(record.pathname)
        while current_path != os.path.dirname(current_path):  # Stop at filesystem root
            if os.path.exists(os.path.join(current_path, 'nemo', '__init__.py')):
                nemo_root = current_path
                break
            current_path = os.path.dirname(current_path)
        
        if nemo_root:
            record.relative_path = os.path.relpath(record.pathname, nemo_root)
        else:
            record.relative_path = record.pathname
        formatted = super().format(record)
        return formatted.replace(record.pathname, record.relative_path)

# class BaseFuncNameFormatter(BaseFormatter):
#     DEFAULT_FORMAT = "%(color)s[%(levelname)1.1s %(asctime)s %(pathname)s:%(lineno)d][%(funcName)s()]%(end_color)s %(message)s"

class BaseFuncNameNeMoFormatterWithRelativePath(BaseNeMoFormatterWithRelativePath):
    DEFAULT_FORMAT = "%(color)s[NeMo %(levelname)1.1s %(asctime)s %(pathname)s:%(lineno)d][%(funcName)s()]%(end_color)s %(message)s"

class DebugFuncNameNeMoFormatterWithRelativePath(BaseNeMoFormatterWithRelativePath):
    DEFAULT_FORMAT = (
        "%(color)s[NeMo %(levelname)1.1s %(asctime)s %(pathname)s:%(lineno)d rank:%(rank)s][%(funcName)s()]%(end_color)s %(message)s"
    )

logger = logging._logger
# Update logger handlers to use function name formatter
if hasattr(logger, 'handlers'):
    for handler in logger.handlers:
        if hasattr(handler, 'formatter') and handler.formatter is not None:
            if isinstance(handler.formatter, BaseNeMoFormatter):
                handler.setFormatter(BaseFuncNameNeMoFormatterWithRelativePath())
            elif isinstance(handler.formatter, DebugNeMoFormatter):
                handler.setFormatter(DebugFuncNameNeMoFormatterWithRelativePath())
# logging.remove_stream_handlers()
# logging.add_stream_handlers(formatter=BaseFuncNameFormatter)

# Set up logging with stream handler and custom formatter
def setup_logging():
    import logging
    # Set default formatter for root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        force=True
    )    

    loggers = [name for name in logging.root.manager.loggerDict]
    for logger_name in loggers:
        # logging.info(logger_name)
        if logger_name == 'nemo_logger':
            continue

        logger = logging.getLogger(logger_name)
        
        # Create stream handler
        stream_handler = logging.StreamHandler()
        
        # Create formatter with the specified format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s')
        stream_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

# Initialize logging for all hierarchical loggers under cosyvoice
setup_logging()


class CosyVoice2AudioDecoder(torch.nn.Module):
    """
    Minimal adapter to use CosyVoice2 Causal Streaming Flow in NeMo training.

    - .flow(batch, device): compute CFM training loss
    - .token2wav(...): decode tokens -> mel via CosyVoice2 flow -> waveform via HiFT (HiFi-GAN)
      Note: We resample 24k output to 22.05k to match existing downstream code.
    - .stream_inference(...): streaming decode with overlap-fade and HiFT cache (24k -> 22.05k)
    
    Naming Conventions:
    -------------------
    Batch-related:
        batch_size (B)      : Total number of samples in batch
        batch_idx          : Index variable for iterating over batch (0 to B-1)
        
    Sample-related (single sample from batch):
        sample_*           : Variables representing a single sample extracted from batch
                            (e.g., sample_token, sample_mel, sample_waveform, sample_embedding)
                            Use instead of *_b suffix for clarity
        
    Token-related:
        token_*            : Speech token variables (prefer full 'token' over 'tok')
        text_token_*       : Text token variables
        num_tokens_*       : Token count/length variables
        
    Time dimensions:
        num_tokens_original    : Original token sequence length (pre-upsampling)
        num_tokens_upsampled   : Token sequence length after upsampling
        num_hidden_frames      : Hidden state temporal dimension from encoder
        num_valid_frames       : Valid (non-padded) frames in sequence
        
    Upsampling state suffixes (for token sequences that may be upsampled):
        *_original             : Original/pre-upsampled state (e.g., token_emb_original, token_ids_original)
        *_upsampled            : Post-upsampled state (e.g., token_ids_upsampled, block_size_upsampled)
        
    Embedding suffixes (for complete sequence embeddings):
        *_emb_full             : Complete sequence embeddings (entire sequence, not chunked)
        Examples:
            token_emb_full          : Complete upsampled token embeddings for entire sequence
            text_token_emb_full     : Complete text token embeddings for entire text sequence
            
    Hidden states & features:
        hidden_*           : Encoder hidden states (e.g., hidden_encoded, hidden_chunk)
        feat_*             : Mel-spectrogram features
        embedding_*        : Speaker/text embeddings
        
    Masks:
        *_mask             : Attention/padding masks (suffix convention)
    """

    # Audio/Feature dimensions
    MEL_DIM = 80
    SPEAKER_EMBEDDING_DIM = 192
    DEFAULT_CONTEXT_DIM = 512 # Attention dim is 512 per CosyVoice2 encoder input size
    
    # Sample rates
    OUTPUT_SAMPLE_RATE = 22050
    DEFAULT_COSYVOICE2_SAMPLE_RATE = 24000
    
    # Token-mel alignment
    DEFAULT_TOKEN_MEL_RATIO = 4.0 # 50fps / 4 = 12.5 fps; one token -> 4 mel frames in CosyVoice2
    ENCODER_UPSAMPLE_FACTOR = 2.0 # Upsample factor from CosyVoice2 encoder to CosyVoice2 flow
    
    # Vocoder settings
    MEL_HOP_LENGTH = 480 # 24k / 480 = 50 fps; one mel frame -> 480 waveform samples in HiFT
    
    # Text/token settings
    DEFAULT_TEXT_VOCAB_SIZE = 32000

    def __init__(
        self,
        cos2_config_path: str,
        device: str = "cuda",
        fallback_flow_dir: Optional[str] = None,
        cos2_config_override: Optional[Dict] = None,
        warmstart_config: Optional[Dict] = None,
        stream_train_prob: Optional[float] = None,
        use_text_context_train: Optional[bool] = None,
        token_overlap: Optional[int] = None,
        is_debug: Optional[bool] = None,
        print_per_n_chunk: Optional[int] = None,
        stream_stride: Optional[int] = None,
        stream_train_first_block_random: Optional[bool] = None,
        stream_fixed_window_pad: Optional[bool] = None
    ):
        """
        Initialize CosyVoice2AudioDecoder with configuration and environment setup.
        
        Parameters:
            cos2_config_path: Path to CosyVoice2 config file
            device: Device to use for training
            fallback_flow_dir: Directory containing fallback flow model
            cos2_config_override: Override CosyVoice2 config dictionary
            warmstart_config: Warmstart configuration dictionary
            stream_train_prob: Probability of training in streaming mode
            use_text_context_train: Whether to use text context in training
            token_overlap: Token overlap length
            is_debug: Debug mode toggle
            print_per_n_chunk: Print frequency for streaming chunks
            stream_stride: Sliding window stride for streaming
            stream_train_first_block_random: Whether to randomize first chunk length
            stream_fixed_window_pad: Whether to pad last window to fixed chunk size
        """
        # Initialize base class
        super().__init__()
        self.device = device
        self._cos2_config_override = cos2_config_override or {}
        self._warm_cfg = warmstart_config or {}
        self._stream_train_prob = 0.5 if stream_train_prob is None else float(stream_train_prob)
        self._use_text_context_train = bool(use_text_context_train) if use_text_context_train is not None else False
        self._token_overlap_len_cfg = 0 if token_overlap is None else int(token_overlap)
        self._fixed_window_pad = bool(stream_fixed_window_pad) if stream_fixed_window_pad is not None else False

        # debug control
        self._step = 0
        self._debug_every = int(os.environ.get("COS2_DEBUG_EVERY", "200"))
        # Validation-time debug toggle (overridable by constructor args)
        self._val_debug = bool(is_debug) if is_debug is not None else bool(int(os.environ.get("COS2_VAL_DEBUG", "0")))
        # Text/token alignment debug toggle
        self._debug_text_align = bool(int(os.environ.get("COS2_DEBUG_TEXT_ALIGN", "0")))
        # Cross-attn debug toggle
        self._debug_xattn = bool(int(os.environ.get("COS2_DEBUG_XATTN", "0")))
        # In-batch chunk debug sampling interval (print every K-th chunk); default 10
        self._print_every_k_in_batch = int(os.environ.get("COS2_PRINT_EVERY_K_IN_BATCH", "10"))
        # Blocks debug toggle: print stats around speech/text transformer blocks
        self._debug_blocks = bool(int(os.environ.get("COS2_DEBUG_BLOCKS", "0")))
        # Streaming print frequency control
        self._print_per_n_chunk = int(print_per_n_chunk) if print_per_n_chunk is not None else int(os.environ.get("COS2_PRINT_PER_N_CHUNK", "100"))


        # Ensure CosyVoice2 repo takes precedence over NeMo's bundled cosyvoice to resolve imports like
        # 'cosyvoice.flow.flow.CausalMaskedDiffWithXvec'.
        def _infer_cosyvoice2_root(cos2_config_path: str) -> Optional[str]:
            # 从cos2_config_path推断cosyvoice仓库的root (e.g., /path/to/CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml -> /path/to/CosyVoice)
            cosyvoice2_root = None

            if os.path.isfile(cos2_config_path):
                # Go up from config path: conf -> cosyvoice2 -> libritts -> examples -> CosyVoice (repo root)
                config_dir = os.path.dirname(cos2_config_path)  # .../conf
                cosyvoice2_dir = os.path.dirname(config_dir)    # .../cosyvoice2
                libritts_dir = os.path.dirname(cosyvoice2_dir)  # .../libritts
                examples_dir = os.path.dirname(libritts_dir)    # .../examples
                potential_root = os.path.dirname(examples_dir)  # .../CosyVoice
                # Verify this looks like a CosyVoice repo by checking for key files/dirs
                # Accept repo root if it contains the 'cosyvoice' package directory; do not require setup.py
                if os.path.isdir(os.path.join(potential_root, "cosyvoice")):
                    cosyvoice2_root = potential_root

            # Fallback to env var if inference failed
            if not cosyvoice2_root:
                cosyvoice2_root = os.environ.get("COS2_REPO_ROOT")

            return cosyvoice2_root

        cosyvoice2_root = _infer_cosyvoice2_root(cos2_config_path)
        if cosyvoice2_root and os.path.isdir(cosyvoice2_root) and cosyvoice2_root not in sys.path:
            sys.path.insert(0, cosyvoice2_root)
        from cosyvoice.transformer.attention import MultiHeadedAttention

        if load_hyperpyyaml is None:
            raise ImportError("hyperpyyaml is required to load CosyVoice2 yaml configs.")

        if not os.path.isfile(cos2_config_path):
            raise FileNotFoundError(f"CosyVoice2 config not found: {cos2_config_path}")

        # Load flow from CosyVoice2 yaml
        with open(cos2_config_path, "r") as f:
            # Only build 'flow' to avoid initializing LLM, vocoder, tokenizer & dataset pipeline components
            overrides = {
                "llm": None,
                "hift": None,
                "hifigan": None,
                # dataset/tokenizer related
                "parquet_opener": None,
                "get_tokenizer": None,
                "tokenize": None,
                "filter": None,
                "resample": None,
                "truncate": None,
                "feat_extractor": None,
                "compute_fbank": None,
                "compute_f0": None,
                "parse_embedding": None,
                "shuffle": None,
                "sort": None,
                "batch": None,
                "padding": None,
                "data_pipeline": None,
                "data_pipeline_gan": None,
                "mel_spec_transform1": None,
                "mel_spec_transform": None,
            }
            overrides.update(self._cos2_config_override)
            configs = load_hyperpyyaml(f, overrides=overrides, overrides_must_match=False)
        # CosyVoice2 config exposes 'flow' module consistent with CausalMaskedDiffWithXvec
        self.cos2_flow = configs["flow"]
        # remember config path for later HiFT lazy init

        self._cos2_config_path = cos2_config_path
        # sample rate used by CosyVoice2

        # Cross-text attention (optional; streaming only). Initialize defaults first
        self._use_cross_text_attn = bool(int(os.environ.get("COS2_USE_CROSS_TEXT_ATTN", "0")))
        self._cross_text_heads = int(os.environ.get("COS2_CROSS_TEXT_HEADS", "4"))
        self._cross_text_dropout = float(os.environ.get("COS2_CROSS_TEXT_DROPOUT", "0.0"))
        self._cross_q_pool = os.environ.get("COS2_CROSS_Q_POOL", "mean")  # 'mean' or 'last'

        # Override from YAML config if present
        self._use_cross_text_attn = bool(configs.get('use_cross_text_attn', self._use_cross_text_attn))
        self._cross_text_heads = int(configs.get('cross_text_attn_heads', self._cross_text_heads))
        self._cross_text_dropout = float(configs.get('cross_text_attn_dropout', self._cross_text_dropout))
        self._cross_q_pool = str(configs.get('cross_text_q_pool', self._cross_q_pool))

        # sliding-window stride (in tokens), default 0 -> fallback to chunk_size stepping
        self._stream_stride = int(configs.get('stream_stride', 0))
        if stream_stride is not None and int(stream_stride) > 0:
            self._stream_stride = int(stream_stride)

        # Training-time augmentation: randomize first chunk length (option 2)
        self._stream_train_first_block_random = bool(configs.get('stream_train_first_block_random', False))
        if stream_train_first_block_random is not None:
            self._stream_train_first_block_random = bool(stream_train_first_block_random)

        # The pre_lookahead_len (L_ctx) controls expected context length to encoder
        self._L_ctx_default = int(getattr(self.cos2_flow.encoder.pre_lookahead_layer, 'pre_lookahead_len', 4))
        # Attention dim is 512 per CosyVoice2 encoder input size
        self._ctx_dim = self.DEFAULT_CONTEXT_DIM

        # Text embedding for context-based lookahead (initialized here; trained later if needed)
        # infer text vocab size from env (default 32000)
        text_vocab_size = int(os.environ.get("COS2_TEXT_VOCAB_SIZE", str(self.DEFAULT_TEXT_VOCAB_SIZE)))
        self._text_vocab_size = text_vocab_size
        self.text_context_emb = torch.nn.Embedding(self._text_vocab_size, self._ctx_dim)

        # Create cross-attention layer once with all configs resolved
        self.cross_text_attn = MultiHeadedAttention(
            n_head=self._cross_text_heads,
            n_feat=self._ctx_dim,
            dropout_rate=self._cross_text_dropout,
        )
        # FFN after cross-attn (Pre-LN + Residual)
        self._cross_ffn_hidden = int(configs.get('cross_text_ffn_hidden', 2048))
        self._cross_ffn_dropout = float(configs.get('cross_text_ffn_dropout', 0.1))
        self.cross_ffn = torch.nn.Sequential(
            torch.nn.Linear(self._ctx_dim, self._cross_ffn_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self._cross_ffn_dropout),
            torch.nn.Linear(self._cross_ffn_hidden, self._ctx_dim),
            torch.nn.Dropout(self._cross_ffn_dropout),
        )
        # Expand [B,1,512] -> [B,L_ctx,512]
        self.cross_expand = torch.nn.Linear(self._ctx_dim, self._L_ctx_default * self._ctx_dim)
        self.cross_ln = torch.nn.LayerNorm(self._ctx_dim)
        # Context aggregation from full Q sequence to L_ctx via learned queries (no pooling)
        self.q2ctx_attn = MultiHeadedAttention(
            n_head=self._cross_text_heads,
            n_feat=self._ctx_dim,
            dropout_rate=self._cross_text_dropout,
        )
        self.ctx_queries = torch.nn.Parameter(torch.randn(self._L_ctx_default, self._ctx_dim) * 0.02)

        # Scheme-1 blocks: lightweight self-attn+FFN on speech (Q) and text (KV)
        self._speech_sa_heads = int(configs.get('speech_sa_heads', max(2, self._cross_text_heads // 2)))
        self._speech_sa_dropout = float(configs.get('speech_sa_dropout', 0.0))
        self._speech_ffn_hidden = int(configs.get('speech_ffn_hidden', self._cross_ffn_hidden))
        self._speech_ffn_dropout = float(configs.get('speech_ffn_dropout', self._cross_ffn_dropout))
        self.speech_sa = MultiHeadedAttention(
            n_head=self._speech_sa_heads,
            n_feat=self._ctx_dim,
            dropout_rate=self._speech_sa_dropout,
        )
        self.speech_ln = torch.nn.LayerNorm(self._ctx_dim)
        self.speech_ffn = torch.nn.Sequential(
            torch.nn.Linear(self._ctx_dim, self._speech_ffn_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self._speech_ffn_dropout),
            torch.nn.Linear(self._speech_ffn_hidden, self._ctx_dim),
            torch.nn.Dropout(self._speech_ffn_dropout),
        )

        self._text_sa_heads = int(configs.get('text_sa_heads', self._cross_text_heads))
        self._text_sa_dropout = float(configs.get('text_sa_dropout', 0.0))
        self._text_ffn_hidden = int(configs.get('text_ffn_hidden', self._cross_ffn_hidden))
        self._text_ffn_dropout = float(configs.get('text_ffn_dropout', self._cross_ffn_dropout))
        self.text_sa = MultiHeadedAttention(
            n_head=self._text_sa_heads,
            n_feat=self._ctx_dim,
            dropout_rate=self._text_sa_dropout,
        )
        self.text_ln = torch.nn.LayerNorm(self._ctx_dim)
        self.text_ffn = torch.nn.Sequential(
            torch.nn.Linear(self._ctx_dim, self._text_ffn_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self._text_ffn_dropout),
            torch.nn.Linear(self._text_ffn_hidden, self._ctx_dim),
            torch.nn.Dropout(self._text_ffn_dropout),
        )



        self._cos2_sr = int(configs.get("sample_rate", self.DEFAULT_COSYVOICE2_SAMPLE_RATE)) if isinstance(configs, dict) else self.DEFAULT_COSYVOICE2_SAMPLE_RATE
        # Lazy-initialized vocoder (HiFT)
        self._hift = None
        # Streaming overlap/cache buffers
        self._mel_overlap_dict = defaultdict(lambda: None)
        self._mel_total_len_dict = defaultdict(lambda: 0)  # total mel frames emitted so far per uuid (emitted_total)
        self._mel_model_total_dict = defaultdict(lambda: 0)  # model-side last cumulative T_all per uuid
        self._hift_cache_dict = defaultdict(lambda: None)
        # overlap lengths are derived from token_frame_rate and mel hop
        token_fps = float(getattr(self.cos2_flow, 'input_frame_rate', 25.0))
        # prefer YAML-configured token overlap; default 0 for CosyVoice2
        self._token_overlap_len = int(self._token_overlap_len_cfg)
        # 24k / 480 = 50 fps; one mel frame -> 480 waveform samples in HiFT
        self._mel_hop = self.MEL_HOP_LENGTH
        mel_fps = float(self._cos2_sr) / float(self._mel_hop)
        self._mel_overlap_len = int(self._token_overlap_len / token_fps * mel_fps)
        self._mel_window = np.hamming(max(2 * self._mel_overlap_len, 2))
        # hift cache: cache 1 mel frame worth of source (480 samples at 24k)
        self._mel_cache_len = 1
        self._source_cache_len = int(self._mel_cache_len * self._mel_hop)
        # Warm-start flow parameters if possible (after _cos2_config_path/_hift are set)
        self._maybe_warm_start_flows()
        # Do not rely on stored self.device later; Lightning will move modules.
        # We avoid forcing device here; use runtime param device from model/params instead.
        # self.cos2_flow.to(self.device)

        # Optional: set up fallback inference path using legacy AudioDecoder for token2wav
        # Disabled by default to avoid constructing full LLM from CosyVoice configs.
        # Enable only when explicitly requested via env COS2_ENABLE_FALLBACK=1 and a proper legacy config is provided.
        self._fallback_infer = None
        self._fallback_dir = fallback_flow_dir
        enable_fb = os.environ.get("COS2_ENABLE_FALLBACK", "0") == "1"
        if enable_fb and self._fallback_dir and os.path.isdir(self._fallback_dir):
            cfg_path = os.path.join(self._fallback_dir, "config.yaml")
            flow_ckpt = os.path.join(self._fallback_dir, "flow.pt")
            hift_ckpt = os.path.join(self._fallback_dir, "hift.pt")
            if os.path.isfile(cfg_path):
                from nemo.collections.speechlm2.modules.flow_inference import AudioDecoder as LegacyAudioDecoder
                self._fallback_infer = LegacyAudioDecoder(cfg_path, flow_ckpt, hift_ckpt, device=self.device)

    def _lazy_init_hift(self):
        if self._hift is not None:
            return
        # Build only HiFT from CosyVoice2 yaml
        with open(self._cos2_config_path, "r") as f:
            overrides = {
                "llm": None,
                "flow": None,
                "hifigan": None,  # not needed for inference
                # disable dataset/tokenizer/pipelines
                "parquet_opener": None,
                "get_tokenizer": None,
                "tokenize": None,
                "filter": None,
                "resample": None,
                "truncate": None,
                "feat_extractor": None,
                "compute_fbank": None,
                "compute_f0": None,
                "parse_embedding": None,
                "shuffle": None,
                "sort": None,
                "batch": None,
                "padding": None,
                "data_pipeline": None,
                "data_pipeline_gan": None,
                "mel_spec_transform1": None,
                "mel_spec_transform": None,
            }
            cfg = load_hyperpyyaml(f, overrides=overrides, overrides_must_match=False)
        self._hift = cfg["hift"].to(self.device)
        # refresh sample rate if available
        self._cos2_sr = int(cfg.get("sample_rate", self._cos2_sr))

        # Optionally load HiFT checkpoint for inference quality
        hift_ckpt = os.environ.get("COS2_HIFT_CKPT", None)
        if hift_ckpt and os.path.isfile(hift_ckpt):
            ckpt = torch.load(hift_ckpt, map_location=self.device)
            sd = ckpt.get("state_dict", ckpt)
            # Strip common wrappers
            def _maybe_strip_prefix(state_dict, prefix):
                if all(k.startswith(prefix) for k in state_dict.keys()):
                    return {k[len(prefix):]: v for k, v in state_dict.items()}
                return state_dict
            for pref in ("module.", "generator."):
                sd = _maybe_strip_prefix(sd, pref)
            res = self._hift.load_state_dict(sd, strict=False)
            missing = getattr(res, 'missing_keys', [])
            unexpected = getattr(res, 'unexpected_keys', [])
            logging.info(f"[cos2.hift] loaded '{hift_ckpt}' missing={len(missing)} unexpected={len(unexpected)} sr={self._cos2_sr}")
        else:
            if self._val_debug:
                logging.info("[cos2.hift] no COS2_HIFT_CKPT provided; using randomly initialized HiFT (may sound like noise)")

    def _maybe_warm_start_flows(self):
        """Warm-start CosyVoice2 flow with:
        1) Optional CosyVoice2 pretrained main body (controlled by cfg: warmstart.transfer.cos2_main_body)
        2) Optional overlay of GLM4-Voice selected layers (controlled by cfg: warmstart.transfer.*)
        Paths can be overridden via env:
            - COS2_PRETRAINED_DIR (default: /data/workspace/cache/HFCACHE/cosyvoice/CosyVoice2-0.5B)
            - GLM4_PRETRAINED_DIR (default: /data/workspace/cache/HFCACHE/glm-4-voice-decoder)
        """
        def _strip_prefix(sd, prefixes=("module.",)):
            if not isinstance(sd, dict):
                return sd
            keys = list(sd.keys())
            if all(any(k.startswith(p) for p in prefixes) for k in keys):
                out = {}
                for k, v in sd.items():
                    for p in prefixes:
                        if k.startswith(p):
                            k = k[len(p):]
                            break
                    out[k] = v
                return out
            return sd

        def _filtered_load(module: torch.nn.Module, state_dict: dict):
            msd = module.state_dict()
            keep = {}
            for k, v in state_dict.items():
                if k in msd and msd[k].shape == v.shape:
                    keep[k] = v
            missing, unexpected = module.load_state_dict(keep, strict=False)
            return keep, missing, unexpected

        # 1) Load CosyVoice2 main body if available
        cfg_cos2_dir = None
        cfg_glm4_dir = None
        if isinstance(self._warm_cfg, dict):
            cfg_cos2_dir = self._warm_cfg.get("cos2_dir") or ((self._warm_cfg.get("pretrained_dirs") or {}).get("cos2_dir") if isinstance(self._warm_cfg.get("pretrained_dirs"), dict) else None)
            cfg_glm4_dir = self._warm_cfg.get("glm4_dir") or ((self._warm_cfg.get("pretrained_dirs") or {}).get("glm4_dir") if isinstance(self._warm_cfg.get("pretrained_dirs"), dict) else None)
        cos2_dir = cfg_cos2_dir or os.environ.get("COS2_PRETRAINED_DIR")
        glm4_dir = cfg_glm4_dir or os.environ.get("GLM4_PRETRAINED_DIR")
        cos2_flow_pt = os.path.join(cos2_dir, "flow.pt") if cos2_dir else None
        glm4_flow_pt = os.path.join(glm4_dir, "flow.pt") if glm4_dir else None

        transfer_cfg = (self._warm_cfg.get("transfer") or {}) if isinstance(self._warm_cfg, dict) else {}

        # 1) CosyVoice2 main body (encoder/decoder etc.)
        cos2_sd = None
        if transfer_cfg.get("cos2_main_body", True):
            if os.path.isfile(cos2_flow_pt):
                cos2_sd = torch.load(cos2_flow_pt, map_location="cpu")
                cos2_sd = cos2_sd.get("state_dict", cos2_sd)
                cos2_sd = _strip_prefix(cos2_sd, prefixes=("module.", "flow."))
                kept, missing, unexpected = _filtered_load(self.cos2_flow, cos2_sd)
                # Allow missing input_embedding when vocab size differs; it will be overlaid from GLM4.
                miss_names = set(missing) if isinstance(missing, (list, tuple, set)) else set()
                unexp_names = set(unexpected) if isinstance(unexpected, (list, tuple, set)) else set()
                # Allow kernel-size-changed conv when pre_lookahead_len differs from checkpoint
                allowed_missing = {
                    "input_embedding.weight",
                    "encoder.pre_lookahead_layer.conv1.weight",
                }
                if unexp_names or (miss_names - allowed_missing):
                    raise RuntimeError(
                        f"Cos2 main-body load mismatch: missing={list(miss_names)} unexpected={list(unexp_names)}"
                    )
                if miss_names:
                    logging.info(f"[warm] loaded CosyVoice2 main body (ignored missing keys: {list(miss_names)}) from {cos2_flow_pt}: kept={len(kept)}")
                else:
                    logging.info(f"[warm] loaded CosyVoice2 flow main body from {cos2_flow_pt}: kept={len(kept)}")
            else:
                logging.info(f"[warm] skip: CosyVoice2 flow checkpoint not found (cos2_dir={cos2_dir})")
        else:
            logging.info("[warm] skip loading CosyVoice2 main body due to cfg.transfer.cos2_main_body=false")

        # 2) Overlay GLM4-Voice selected layers
        if os.path.isfile(glm4_flow_pt):
            gsd = torch.load(glm4_flow_pt, map_location="cpu")
            gsd = gsd.get("state_dict", gsd)
            gsd = _strip_prefix(gsd, prefixes=("module.", "flow."))
            msd = self.cos2_flow.state_dict()
            picks = []
            if transfer_cfg.get("input_embedding", True):
                picks.append("input_embedding.weight")
            if transfer_cfg.get("spk_affine", True):
                picks += ["spk_embed_affine_layer.weight", "spk_embed_affine_layer.bias"]
            # encoder_proj 覆盖优先级：如果你想强制恢复 CosyVoice2 的 encoder_proj，可在 transfer.encoder_proj=false；
            # 若 transfer.encoder_proj=true，仍以 GLM4 覆盖。
            if transfer_cfg.get("encoder_proj", False):
                picks += ["encoder_proj.weight", "encoder_proj.bias"]
            copied = 0
            for k in picks:
                if k in gsd and k in msd and gsd[k].shape == msd[k].shape:
                    msd[k] = gsd[k]
                    copied += 1
                else:
                    raise RuntimeError(f"Shape mismatch or missing when overlay {k}: glm4={gsd.get(k, None) and tuple(gsd[k].shape)} cos2={msd.get(k, None) and tuple(msd[k].shape)}")
            self.cos2_flow.load_state_dict(msd, strict=False)
            logging.info(f"[warm] overlaid GLM4 layers from {glm4_flow_pt}: copied={copied}/{len(picks)}")
        else:
            logging.info(f"[warm] skip: GLM4 flow checkpoint not found (glm4_dir={glm4_dir})")

        # 如果需要显式地从 CosyVoice2 主体权重里“再恢复一次 encoder_proj”，加一个后置覆盖（在未开启 GLM4 覆盖时才执行）
        if cos2_sd is not None and not transfer_cfg.get("encoder_proj", False):
            msd = self.cos2_flow.state_dict()
            for k in ("encoder_proj.weight", "encoder_proj.bias"):
                if k in cos2_sd and k in msd and cos2_sd[k].shape == msd[k].shape:
                    msd[k] = cos2_sd[k]
            self.cos2_flow.load_state_dict(msd, strict=False)
            logging.info("[warm] ensured CosyVoice2 encoder_proj restored from main body")

        if self._hift is not None:
            return
        # Build only HiFT from CosyVoice2 yaml
        with open(self._cos2_config_path, "r") as f:
            overrides = {
                "llm": None,
                "flow": None,
                "hifigan": None,  # not needed for inference
                # disable dataset/tokenizer/pipelines
                "parquet_opener": None,
                "get_tokenizer": None,
                "tokenize": None,
                "filter": None,
                "resample": None,
                "truncate": None,
                "feat_extractor": None,
                "compute_fbank": None,
                "compute_f0": None,
                "parse_embedding": None,
                "shuffle": None,
                "sort": None,
                "batch": None,
                "padding": None,
                "data_pipeline": None,
                "data_pipeline_gan": None,
                "mel_spec_transform1": None,
                "mel_spec_transform": None,
            }
            cfg = load_hyperpyyaml(f, overrides=overrides, overrides_must_match=False)
        self._hift = cfg["hift"].to(self.device)
        # refresh sample rate if available
        self._cos2_sr = int(cfg.get("sample_rate", self._cos2_sr))

        # Optionally load HiFT checkpoint for inference quality
        hift_ckpt = os.environ.get("COS2_HIFT_CKPT", None)
        if hift_ckpt and os.path.isfile(hift_ckpt):
            ckpt = torch.load(hift_ckpt, map_location=self.device)
            sd = ckpt.get("state_dict", ckpt)
            # Strip common wrappers
            def _maybe_strip_prefix(state_dict, prefix):
                if all(k.startswith(prefix) for k in state_dict.keys()):
                    return {k[len(prefix):]: v for k, v in state_dict.items()}
                return state_dict
            for pref in ("module.", "generator."):
                sd = _maybe_strip_prefix(sd, pref)
            res = self._hift.load_state_dict(sd, strict=False)
            if self._val_debug or True:
                missing = getattr(res, 'missing_keys', [])
                unexpected = getattr(res, 'unexpected_keys', [])
                logging.info(f"[cos2.hift] loaded '{hift_ckpt}' missing={len(missing)} unexpected={len(unexpected)} sr={self._cos2_sr}")
        else:
            if self._val_debug:
                logging.info("[cos2.hift] no COS2_HIFT_CKPT provided; using randomly initialized HiFT (may sound like noise)")

    def _ensure_device(self, device):
        """Prepare inputs from batch and ensure all modules are on the correct device."""
        # Ensure cos2_flow on correct device
        pdev = next(self.cos2_flow.parameters()).device
        if pdev != device:
            self.cos2_flow.to(device)
        # Ensure adapter submodules on correct device
        self.text_context_emb = self.text_context_emb.to(device)
        self.cross_text_attn = self.cross_text_attn.to(device)
        self.q2ctx_attn = self.q2ctx_attn.to(device)
        self.cross_expand = self.cross_expand.to(device)
        self.cross_ln = self.cross_ln.to(device)
        self.cross_ffn = self.cross_ffn.to(device)
        # scheme-1 blocks to device
        self.speech_sa = self.speech_sa.to(device)
        self.speech_ln = self.speech_ln.to(device)
        self.speech_ffn = self.speech_ffn.to(device)
        self.text_sa = self.text_sa.to(device)
        self.text_ln = self.text_ln.to(device)
        self.text_ffn = self.text_ffn.to(device)

    def _determine_chunk_block_size(self):
        """Determine chunking block size in original token units.
        
        Returns:
            block_size (int): Chunking block size in original token units
        """
        block_size = None
        bs = getattr(self.cos2_flow.encoder, 'static_chunk_size', None)
        if isinstance(bs, (int, float)) and bs > 0:
            block_size = int(bs)
        if not block_size:
            bs = getattr(self.cos2_flow, 'static_chunk_size', None)
            if isinstance(bs, (int, float)) and bs > 0:
                block_size = int(bs)
        if not block_size:
            block_size = int(os.environ.get('COS2_STATIC_CHUNK_SIZE', '2'))
        if block_size <= 0:
            block_size = 2  # safety fallback
        return block_size

    def _compute_upsample_factor(self):
        """Compute upsampling factor for token alignment.
        
        Returns:
            upsample_factor (int): Upsampling factor (typically 2)
        """
        tmr = float(getattr(self.cos2_flow, 'token_mel_ratio', self.DEFAULT_TOKEN_MEL_RATIO))
        enc_up = self.ENCODER_UPSAMPLE_FACTOR
        upsample_factor = max(1, int(round(tmr / enc_up)))
        return upsample_factor

    def flow(self, batch: Dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """
        计算 CosyVoice2 因果流（Causal Flow, CFM）的训练损失，并显式对齐时间轴。

        训练总体流程（含流式/非流式两种路径）：
          1) 将离散语音 token（speech_token）嵌入，按概率走"流式切块编码"或"整段一次编码"。
          2) 对编码器输出 hidden 进行 encoder_proj 得到 [B, T_h, 80]，并产出时长掩码 hidden_mask。
          3) 将 GT 梅尔谱（speech_feat）沿时间插值到 T_h，得到 x1 = [B, 80, T_h]。
          4) 构造与 CosyVoice2 原训练一致的条件 cond（最多 30% 的随机前缀），同样对齐到 T_h。
          5) 调用 decoder.compute_loss(x1, mask, h^T, embedding, cond, streaming) 计算损失。

        输入 batch 字段与形状：
          - speech_token:      [B, T_tok]       离散语音 token 序列（WhisperVQ 12.5Hz）
          - speech_token_len:  [B]              每条样本的 token 有效长度
          - speech_feat:       [B, 80, T_feat] 或 [B, T_feat, 80]   目标梅尔谱
          - speech_feat_len:   [B]              梅尔帧有效长度（主要用于日志/校验）
          - embedding:         [B, 192]         说话人嵌入；本函数内先归一化再经 spk_embed_affine_layer 投影
          - text_tokens:       [B, T_txt] 可选  文本 token（用于流式 cross-attention 上下文）
          - text_token_len:    [B]       可选  文本 token 有效长度

        关键参数/成员（仅与本函数相关）：
          - _stream_train_prob: float  进入流式训练分支的概率（其余样本走非流式）
          - _use_text_context_train: 是否使用text context in training
          - token_mel_ratio (tmr):     每个 token 约对应的 mel 帧数（默认 4.0）
          - enc_up:                    编码器时间上采样因子（UpsampleConformerEncoder，固定 2.0）
          - upsample_factor:           round(tmr/enc_up)，通常为 2；决定 token 时间与 encoder 时间对齐关系
          - L_ctx:                     上下文长度槽位（等价于原 pre_lookahead_len），供 encoder(context=...) 使用
          - _use_cross_text_attn:      是否用文本跨注意力来替代 pre_lookahead（True 时启用 cross-attn）
          - _cross_q_pool:             Q 的池化方式，'mean' 或 'last'（从当前 chunk 的 token 表征池化而来）
          - static_chunk_size:         流式块长（单位：原始 token）；从 encoder/static_chunk_size 或环境变量获取

        流式（streaming=True）时每个样本的中间变量：
          - num_tokens_original:    [B]             原始 token 长度（未上采样）
          - token_ids_upsampled: [1, T_upsampled]   上采样后 token 序列（按 upsample_factor 重复）
          - num_tokens_upsampled:    int             上采样后的总步数（可能包含 pad）
          - block_size_upsampled:int             一个 chunk 在"上采样后时间"里的长度（= block_size * upsample_factor）
          - chunk_starts:  List[int]       每个 chunk 的起始下标（上采样后坐标系）
          - token_slices:    List[Tensor]    各 chunk 的变长 token 嵌入切片（未 pad）
          - token_lens:      List[int]       各 chunk 的有效长度（去掉越界 pad 后）
          - token_batch:     [N_chunk, Lmax, D]  将 token_slices pad 后的批处理张量
          - len_vec:       [N_chunk]       每个 chunk 的有效长度向量
          - token_mask:      [N_chunk, Lmax, 1]  token 有效位的掩码
          - text_token_emb_full:  [1, T_txt, 512] 文本全序列的嵌入（如果提供 text_tokens）
          - t_ends:        List[int]       每个 chunk 当前可见的"文本历史长度"（随时间推进递增）
          - kv_hist_max:   int             所有 chunk 的最大可见文本长度（用于对齐 KV 的 pad）
          - text_ctx_batch:[N_chunk, L_ctx, 512]  cross-attn 得到的上下文序列（替代 pre_lookahead）
          - hidden_chunks/hidden_chunks_mask:       编码器对 chunk 批的输出与掩码（hidden_chunks_mask: [N_chunk,1,T_enc]）
          - Li_list:       List[int]       各 chunk 编码器的有效步数（按 hidden_chunks_mask 统计）
          - hidden_b/mask_b:    [1, T_total,80]/[1,1,T_total]  将各 chunk 有效部分拼接回单样本时间轴

        非流式（streaming=False）时：
          - 直接对上采样后的整段 token 一次前向，训练阶段不注入任何上下文（ctx_used=False）。

        输出：
          - 返回字典 {'loss': Tensor 标量}

        日志与一致性校验：
          - [cos2.train.mb] 打印流式每样本的 chunk 统计、显存占用、期望/实际步数等
          - [cos2.train]    打印对齐比例：T_h_valid/T_tok(orig/eff) 与 token_mel_ratio/上采样因子的关系
        """
        
        # ============================================================================
        # Helper functions (defined within flow() for encapsulation and readability)
        # ============================================================================
        
        def _infer_device_from_batch(batch_dict, fallback_device):
            """Infer device from batch tensors to align with DDP shard device.
            
            Args:
                batch_dict (dict): The input batch dictionary containing 'speech_token', 'speech_feat', 'embedding'
                fallback_device (torch.device): Default device to use if no tensor is found
                
            Returns:
                self_device (torch.device): The inferred device
            """
            self_device = fallback_device
            if isinstance(batch_dict, dict):
                for k in ("speech_token", "speech_feat", "embedding"):
                    if k in batch_dict and isinstance(batch_dict[k], torch.Tensor):
                        self_device = batch_dict[k].device
                        break
            return self_device
        
        def _prepare_inputs_and_ensure_device(batch_dict, target_device):
            """Prepare inputs from batch and ensure all modules are on the correct device.
            
            Args:
                batch_dict (dict): The input batch dictionary containing 'speech_token', 'speech_token_len', 'speech_feat', 'embedding'
                target_device (torch.device): Target device to move tensors to
                
            Returns:
                tuple: (token, token_len, feat, feat_len, embedding) where:
                    - token: Speech tokens for the batch [B, T_tok]
                    - token_len: Token lengths for each sample in batch [B]
                    - feat: Feature data for conditioning [B, T_feat, 80]
                    - feat_len: Feature lengths for each sample in batch [B]
                    - embedding: Embedding data for decoder [B, D]
            """
            # Prepare inputs
            token = batch_dict['speech_token'].to(target_device)
            token_len = batch_dict['speech_token_len'].to(target_device)
            feat = batch_dict['speech_feat'].to(target_device)
            if feat.ndim == 3 and feat.shape[1] == self.MEL_DIM:  # [B, MEL_DIM, T] -> [B, T, MEL_DIM]
                feat = feat.transpose(1, 2).contiguous()
            feat_len = batch_dict['speech_feat_len'].to(target_device)

            embedding = batch_dict['embedding'].to(target_device)

            # xvec projection
            embedding = F.normalize(embedding, dim=1)
            embedding = self.cos2_flow.spk_embed_affine_layer(embedding)

            self._ensure_device(target_device)
            
            return token, token_len, feat, feat_len, embedding
        
        def _build_cross_attention_text_context(
            text_token_emb_full, chunk_starts_token, chunk_ends_token, token_len_b,
            token_emb_original, num_chunks, kv_hist_max, t_ends, device_target, b_idx
        ):
            """Build text context using cross-attention mechanism for streaming chunks.
            
            Args:
                text_token_emb_full (torch.Tensor): Full text embeddings [1, N, D]
                chunk_starts_token (list): Start indices for each chunk on token axis
                chunk_ends_token (list): End indices for each chunk on token axis
                token_len_b (int): Effective token length for the current sample
                token_emb_original (torch.Tensor): Original token embeddings [1, T_tok, D]
                num_chunks (int): Number of chunks in the batch
                kv_hist_max (int): Maximum visible text length for cross-attention
                t_ends (list): End indices for each chunk on text axis
                device_target (torch.device): Target device for computations
                b_idx (int): Batch index of the current sample

            Variables:
                L_ctx (int): Context length for text context (self._L_ctx_default)
                KV (torch.Tensor): Text embeddings padded to kv_hist_max [N, kv_hist_max, D]
                mask_kv (torch.Tensor): Mask for KV [N, 1, kv_hist_max]
                t_starts (list): Start indices for each chunk on original token axis
                end_tok_list (list): End indices for each chunk on original token axis
                L_hist_max_tok (int): Maximum length of speech history in original token units
                s_hist_batch (torch.Tensor): Speech history batch [N, L_hist_max_tok, D]
                
            Returns:
                text_ctx_batch (torch.Tensor): Text context batch [N, L_ctx, D]
            """
            from cosyvoice.utils.mask import make_pad_mask
            
            L_ctx = self._L_ctx_default
            # Prepare KV (text embeddings) padded/truncated to kv_hist_max; per-chunk variable visible length via mask_kv
            KV = text_token_emb_full[:, :kv_hist_max].repeat(num_chunks, 1, 1)  # [N, kv_hist_max, D]
            mask_kv = torch.zeros(num_chunks, 1, kv_hist_max, dtype=torch.bool, device=device_target)
            for i, te in enumerate(t_ends):
                t_end_i = int(min(int(te), kv_hist_max))
                if t_end_i > 0:
                    KV[i, :t_end_i] = text_token_emb_full[0, :t_end_i]
                    mask_kv[i, 0, :t_end_i] = True

            # Build speech histories per chunk on ORIGINAL token axis (pre-upsampling): use all past + current chunk tokens
            N = len(chunk_starts_token)
            # Start/end on original axis come directly from chunk_starts_token/chunk_ends_token
            t_starts = [int(s) for s in chunk_starts_token]
            end_token_list = [int(min(int(e), int(token_len_b))) for e in chunk_ends_token]
            L_hist_max_token = max(end_token_list) if len(end_token_list) > 0 else 1
            
            # Assemble history batches padded to L_hist_max_token
            s_hist_batch = token_emb_original.new_zeros((N, L_hist_max_token, token_emb_original.shape[-1]))
            for i in range(N):
                end_token = end_token_list[i]
                if end_token > 0:
                    s_hist_batch[i, :end_token] = token_emb_original[0, :end_token]
            
            # Causal mask over speech history (allow full look-back) on original axis
            mask_s_hist = torch.zeros(N, L_hist_max_token, L_hist_max_token, dtype=torch.bool, device=token_emb_original.device)
            for i in range(N):
                e = end_token_list[i]
                if e > 0:
                    mask_s_hist[i, :e, :e] = torch.tril(torch.ones((e, e), dtype=torch.bool, device=token_emb_original.device))
            
            # Self-attn + FFN over speech histories
            s_hist_attn, _ = self.speech_sa(query=s_hist_batch, key=s_hist_batch, value=s_hist_batch, mask=mask_s_hist)
            s_hist = self.speech_ln(s_hist_attn)
            
            # Debug: stats before and after speech FFN (first batch only, periodic)
            if getattr(self, '_debug_blocks', False) and (b_idx == 0) and ((int(self._step) % max(self._print_per_n_chunk, 1)) == 0):
                try:
                    _s_pre = s_hist
                    _m_pre = float(_s_pre.mean().item()); _sd_pre = float(_s_pre.std(unbiased=False).item())
                except Exception:
                    _m_pre = 0.0; _sd_pre = 0.0
            s_hist = s_hist + self.speech_ffn(s_hist)
            if getattr(self, '_debug_blocks', False) and (b_idx == 0) and ((int(self._step) % max(self._print_per_n_chunk, 1)) == 0):
                try:
                    _m_post = float(s_hist.mean().item()); _sd_post = float(s_hist.std(unbiased=False).item())
                    logging.info(f"[cos2.train.block.speech] b=0 T_hist_max={int(L_hist_max_tok)} mean_pre={_m_pre:.4f} std_pre={_sd_pre:.4f} mean_post={_m_post:.4f} std_post={_sd_post:.4f}")
                except Exception:
                    pass
            
            # Extract current-chunk subrange per row [t_start:t_end) as multi-query Q (original axis), pad to Lq_max
            Lq_list_tok = [max(0, end_tok_list[i] - t_starts[i]) for i in range(N)]
            Lq_max = max(Lq_list_tok) if len(Lq_list_tok) > 0 else 1
            Dq = s_hist.shape[-1]
            s = s_hist.new_zeros((N, Lq_max, Dq))
            for i in range(N):
                Li = int(Lq_list_tok[i])
                if Li <= 0:
                    continue
                s[i, :Li] = s_hist[i, t_starts[i]:end_tok_list[i]]
            
            # mask for Q length (not strictly needed by current cross_attn impl, but useful for clarity)
            mask_qkv = torch.zeros(N, 1, Lq_max, dtype=torch.bool, device=s.device)
            for i in range(N):
                Li = int(Lq_list_tok[i])
                if Li > 0:
                    mask_qkv[i, 0, :Li] = True
            
            # Scheme-1: text-side causal self-attn + FFN
            # Causal self-attn on text (only look-back w.r.t. each position and t_end)
            mask_kv_causal = torch.zeros(num_chunks, kv_hist_max, kv_hist_max, dtype=torch.bool, device=device_target)
            for i, te in enumerate(t_ends):
                t = int(min(int(te), kv_hist_max))
                if t > 0:
                    mask_kv_causal[i, :t, :t] = torch.tril(torch.ones((t, t), dtype=torch.bool, device=device_target))
            kv_attn, _ = self.text_sa(query=KV, key=KV, value=KV, mask=mask_kv_causal)
            kv_ref = self.text_ln(kv_attn)
            
            # Debug: stats before and after text FFN (use first chunk's visible length)
            if getattr(self, '_debug_blocks', False) and (b_idx == 0) and ((int(self._step) % max(self._print_per_n_chunk, 1)) == 0):
                try:
                    _t0 = int(min(int(t_ends[0]), kv_hist_max)) if isinstance(t_ends, (list, tuple)) and len(t_ends) > 0 else kv_hist_max
                    _t0 = max(0, _t0)
                    _kv_pre = kv_ref[0, :_t0] if _t0 > 0 else kv_ref[0:1, :0]
                    _m_pre = float(_kv_pre.mean().item()) if _t0 > 0 else 0.0
                    _sd_pre = float(_kv_pre.std(unbiased=False).item()) if _t0 > 0 else 0.0
                except Exception:
                    _m_pre = 0.0; _sd_pre = 0.0
            kv_ref = kv_ref + self.text_ffn(kv_ref)
            if getattr(self, '_debug_blocks', False) and (b_idx == 0) and ((int(self._step) % max(self._print_per_n_chunk, 1)) == 0):
                try:
                    _kv_post = kv_ref[0, :_t0] if '_t0' in locals() and _t0 > 0 else kv_ref[0:1, :0]
                    _m_post = float(_kv_post.mean().item()) if '_t0' in locals() and _t0 > 0 else 0.0
                    _sd_post = float(_kv_post.std(unbiased=False).item()) if '_t0' in locals() and _t0 > 0 else 0.0
                    logging.info(f"[cos2.train.block.text] b=0 K_max={int(kv_hist_max)} K_vis={int(_t0)} mean_pre={_m_pre:.4f} std_pre={_sd_pre:.4f} mean_post={_m_post:.4f} std_post={_sd_post:.4f}")
                except Exception:
                    pass

            # Cross-attn: multi-query (aligned to chunk length). Then pool last valid to 1 vector per chunk
            attn_out, attn_w = self.cross_text_attn(query=s, key=kv_ref, value=kv_ref, mask=mask_kv)  # [N, Lmax, D]
            
            # Optional debug: inspect attention on chunks spaced every K within the batch
            if getattr(self, '_debug_xattn', False) and (b_idx == 0) and ((int(self._step) % max(self._print_per_n_chunk, 1)) == 0):
                try:
                    # attn_w expected shape [N, H, Q, K] or [N, Q, K]
                    if isinstance(attn_w, torch.Tensor):
                        if attn_w.dim() == 4:
                            aw = attn_w.mean(dim=1)  # [N,Q,K]
                        elif attn_w.dim() == 3:
                            aw = attn_w
                        else:
                            aw = None
                    else:
                        aw = None
                    Kb = int(getattr(self, '_print_every_k_in_batch', 10))
                    for i in range(N):
                        if (i % max(Kb,1) != 0) and (i != N - 1):
                            continue
                        Li = int(Lq_list_tok[i])
                        Ki = int(min(int(t_ends[i]), kv_hist_max))
                        st_tok_i = int(chunk_starts_tok[i]) if 'chunk_starts_tok' in locals() else 0
                        en_tok_i = int(chunk_ends_tok[i]) if 'chunk_ends_tok' in locals() else Li
                        if aw is not None and Li > 0 and Ki > 0:
                            qdim = int(aw.shape[1])
                            kdim = int(aw.shape[2])
                            q_idx = int(max(0, min(Li - 1, qdim - 1)))
                            k_use = int(max(1, min(Ki, kdim)))
                            vec = aw[i, q_idx, :k_use]
                            vec = torch.softmax(vec, dim=-1)
                            k = int(min(3, k_use))
                            vals, idxs = torch.topk(vec, k)
                            idxs = idxs.tolist(); vals = [float(v) for v in vals.tolist()]
                            if q_idx != Li - 1 or k_use != Ki:
                                logging.info(f"[cos2.train.xattn] chunk={i}/{num_chunks} Q_len={Li} K_len={Ki} qidx={q_idx}/{qdim} kdim={kdim} topk_idx={idxs} topk_val={[round(v,4) for v in vals]} (tok=[{st_tok_i}:{en_tok_i}))")
                            else:
                                logging.info(f"[cos2.train.xattn] chunk={i}/{num_chunks} Q_len={Li} K_len={Ki} topk_idx={idxs} topk_val={[round(v,4) for v in vals]} (tok=[{st_tok_i}:{en_tok_i}))")
                        else:
                            logging.info(f"[cos2.train.xattn] chunk={i}/{num_chunks} Q_len={Li} K_len={Ki} (no attn_w) (tok=[{st_tok_i}:{en_tok_i}))")
                except Exception as e:
                    logging.info(f"[cos2.train.xattn] warn: {e}")
            
            # Aggregate full-Q features to L_ctx via learned queries (no temporal pooling)
            q_ctx = self.ctx_queries.to(attn_out.device).unsqueeze(0).expand(num_chunks, -1, -1)  # [N,L_ctx,D]
            # Key mask: valid Q positions only
            key_mask = mask_qkv  # [N,1,Lq_max] with True at valid positions
            ctx_raw, _ = self.q2ctx_attn(query=q_ctx, key=attn_out, value=attn_out, mask=key_mask)  # [N,L_ctx,D]
            x = self.cross_ln(ctx_raw)
            x = x + self.cross_ffn(x)
            text_ctx_batch = x  # [N, L_ctx, D]
            
            return text_ctx_batch
        
        def _build_fallback_text_context(text_token_emb_full, t_ends, L_ctx, device_target):
            """Build simple right-aligned text window per chunk using t_end (fallback when cross-attn disabled).
            
            Args:
                text_token_emb_full (torch.Tensor): Full text embeddings [1, N, D]
                t_ends (list or torch.Tensor): End indices for each chunk [N]
                L_ctx (int): Context length to build
                device_target (torch.device): Target device for computations

            Returns:
                text_ctx_batch (torch.Tensor): Text context batch [N, L_ctx, D]
            """
            ctx_list = []
            for te in t_ends:
                t_end = int(te)
                t_win_emb = text_token_emb_full[:, :t_end]  # [1, N, D] take up to t_end
                if t_win_emb.shape[1] < L_ctx:
                    pad = torch.zeros(1, L_ctx - t_win_emb.shape[1], t_win_emb.shape[2], device=device_target, dtype=t_win_emb.dtype)
                    text_ctx = torch.cat([t_win_emb, pad], dim=1)  # right-pad to L_ctx
                else:
                    text_ctx = t_win_emb[:, -L_ctx:]  # take the most recent L_ctx tokens
                ctx_list.append(text_ctx.squeeze(0))  # [L_ctx, D]
            if len(ctx_list) > 0:
                text_ctx_batch = torch.stack(ctx_list, dim=0)  # [N, L_ctx, D]
            else:
                text_ctx_batch = None
            return text_ctx_batch
        
        def _process_nonstreaming_path(token_data, token_len_data, text_tokens_data, upsample_f, is_streaming, device_target):
            """Process non-streaming training: single pass without text context injection.
            
            Args:
                token_data (torch.Tensor): Token data for conditioning [B, T_tok]
                token_len_data (torch.Tensor): Token lengths for each sample in batch [B]
                text_tokens_data (torch.Tensor): Text tokens for conditioning [B, T_text] or None
                upsample_f (int): Upsampling factor used for token alignment
                is_streaming (bool): Whether in streaming mode
                device_target (torch.device): Target device for computations
                
            Returns:
                tuple: (hidden, hidden_mask) where:
                    - hidden (torch.Tensor): Encoded hidden states from encoder [B, T_h, 80]
                    - hidden_mask (torch.Tensor): Encoder output masks [B, 1, T_h] or similar
            """
            from cosyvoice.utils.mask import make_pad_mask
            
            # upsample entire sequence if needed
            if upsample_f > 1:
                token_data = token_data.repeat_interleave(upsample_f, dim=1)
                token_len_data = token_len_data * upsample_f
            
            # token embedding with padding mask
            token_mask = (~make_pad_mask(token_len_data)).float().unsqueeze(-1).to(device_target)
            token_data = torch.clamp(token_data, min=0)
            token_data = self.cos2_flow.input_embedding(token_data) * token_mask
            
            # Non-streaming: strictly no context injection (pre_lookahead disabled in training)
            hidden, hidden_mask = self.cos2_flow.encoder(token_data, token_len_data, streaming=is_streaming)
            hidden = self.cos2_flow.encoder_proj(hidden)  # [B, T_h, 80]
            
            # Lightweight debug: confirm non-streaming path does not use context
            if (int(self._step) % max(self._debug_every, 1)) == 0 and self._val_debug:
                try:
                    b0 = 0
                    num_text_tokens = int(text_tokens_data.shape[1]) if isinstance(text_tokens_data, torch.Tensor) else 0
                    logging.info(
                        f"[cos2.train.nonstream] B={token_data.shape[0]} num_tokens_upsampled={int(token_len_data[b0].item())} num_text_tokens={num_text_tokens} ctx_used=False xattn={self._use_cross_text_attn}"
                    )
                except Exception:
                    pass
            
            return hidden, hidden_mask
        
        def _build_condition_and_compute_loss(hidden_encoded, hidden_encoded_mask, feat_data, feat_len_data, embedding_data, is_streaming):
            """Build partial cond prefix and compute decoder loss.
            
            Args:
                hidden_encoded (torch.Tensor): Encoded hidden states from encoder [B, T_h, D]
                hidden_encoded_mask (torch.Tensor): Encoder output masks [B, 1, T_h] or similar
                feat_data (torch.Tensor): Feature data for conditioning [B, T_feat, 80]
                feat_len_data (torch.Tensor): Feature length for conditioning [B]
                embedding_data (torch.Tensor): Embedding data for decoder
                is_streaming (bool): Whether in streaming mode
                
            Returns:
                tuple: (loss, lengths) where:
                    - loss (torch.Tensor): Computed decoder loss
                    - lengths (torch.Tensor): Sequence lengths [B]
            """
            from cosyvoice.utils.mask import make_pad_mask
            
            # Build partial cond prefix consistent with CosyVoice2 training (<=30% prefix)
            # Then resample to match encoder output length
            num_hidden_frames = hidden_encoded.shape[1]
            feat_dim = feat_data.shape[2] # 80

            # Interpolate feat_data to num_hidden_frames
            feat_len_data = feat_len_data * num_hidden_frames / feat_data.shape[1] # [B]
            feat_data = F.interpolate(feat_data.unsqueeze(dim=1), size=(num_hidden_frames, feat_dim), mode='nearest').squeeze(dim=1) # [B, T_h, 80]

            conds = feat_data.new_zeros(feat_data.shape)  # [B, T_h, 80]
            for i, j in enumerate(feat_len_data):
                if random.random() < 0.5:
                    continue
                index = random.randint(0, int(0.3 * j))
                conds[i, :index] = feat_data[i, :index]

            # Build mask based on encoder masks -> lengths
            if isinstance(hidden_encoded_mask, torch.Tensor):  # [B,1,T_h] bool
                lengths = hidden_encoded_mask.sum(dim=-1).squeeze(1)
                mask = (~make_pad_mask(lengths)).to(hidden_encoded)
            else:
                # fallback: full True mask
                mask = torch.ones(hidden_encoded.shape[0], hidden_encoded.shape[1], dtype=torch.bool, device=hidden_encoded.device)

            # resample feat to T_h (target x1)
            x1 = feat_data.transpose(1, 2).contiguous()  # [B, 80, T_h]

            conds_chw = conds.transpose(1, 2).contiguous()  # [B, 80, T_h]

            # call decoder loss
            loss, _ = self.cos2_flow.decoder.compute_loss(
                x1,
                mask.unsqueeze(1),
                hidden_encoded.transpose(1, 2).contiguous(),
                embedding_data,
                cond=conds_chw,
                streaming=bool(is_streaming),
            )
            
            return loss, lengths
        
        def _print_training_debug_info(
            loss_val, is_streaming, token_len_data, num_tokens_original_data, hidden_encoded, lengths_data, 
            upsample_f, token_data
        ):
            """Print periodic training-time debug information.
            
            Args:
                loss_val (torch.Tensor): The computed loss value
                is_streaming (bool): Whether streaming mode is enabled
                token_len_data (torch.Tensor): Token lengths for each sample in batch
                num_tokens_original_data (torch.Tensor): Original token lengths before upsampling
                hidden_encoded (torch.Tensor): Encoder hidden states, shape [B, T_h, D]
                lengths_data (torch.Tensor): Valid lengths from encoder masks
                upsample_f (int): Upsampling factor used for token alignment
                token_data (torch.Tensor): Token data tensor
                
            Returns:
                None: This function only prints debug information to logs
            """
            if int(self._step) % int(self._debug_every) == 0:
                try:
                    # pick the first sample in batch for concise logging
                    b0 = 0
                    num_tokens_upsampled_current = int(token_len_data[b0].item()) if torch.is_tensor(token_len_data) else int(token_data.shape[1])
                    # original (pre-upsample) length if available
                    try:
                        num_tokens_original0 = int(num_tokens_original_data[b0].item())
                    except Exception:
                        num_tokens_original0 = int(round(num_tokens_upsampled_current / max(upsample_f, 1))) if upsample_f else num_tokens_upsampled_current
                    # raw encoder time and valid time (from mask)
                    num_hidden_frames_raw = int(hidden_encoded.shape[1])
                    try:
                        num_hidden_frames_valid = int(lengths_data[b0].item())
                    except Exception:
                        num_hidden_frames_valid = num_hidden_frames_raw
                    num_feat_frames = num_hidden_frames_raw  # feat is resampled to hidden frames
                    # ratios computed with valid length (more meaningful than raw tensor length)
                    ratio_upsampled = (num_hidden_frames_valid / max(num_tokens_upsampled_current, 1)) if num_tokens_upsampled_current > 0 else 0.0
                    ratio_original = (num_hidden_frames_valid / max(num_tokens_original0, 1)) if num_tokens_original0 > 0 else 0.0
                    token_fps = getattr(self.cos2_flow, 'input_frame_rate', 'NA')
                    tmr = getattr(self.cos2_flow, 'token_mel_ratio', 'NA')
                    try:
                        _loss_val = float(loss_val.detach().item())
                    except Exception:
                        _loss_val = float('nan')
                    logging.info(
                        f"[cos2.train] step={int(self._step)} streaming={bool(is_streaming)} num_tokens_original={num_tokens_original0} num_tokens_upsampled={num_tokens_upsampled_current} num_hidden_raw={num_hidden_frames_raw} num_hidden_valid={num_hidden_frames_valid} num_feat={num_feat_frames} valid/original={ratio_original:.3f} valid/upsampled={ratio_upsampled:.3f} loss={_loss_val:.6f} cfg: token_fps={token_fps} token_mel_ratio={tmr} up_factor={upsample_f}"
                    )
                except Exception:
                    pass
        
        # ============================================================================
        # Streaming-specific helper functions
        # ============================================================================
        
        def _extract_sample_tokens_and_text(batch_dict, b_idx, text_tokens_all, num_tokens_original_all, batch_size, device_target):
            """Extract tokens and text for a single sample from the batch.
            
            Args:
                batch_dict (dict): The input batch dictionary containing 'speech_token' and other keys
                b_idx (int): The batch index of the sample to extract (0-based)
                text_tokens_all (torch.Tensor or None): All text tokens for the batch, shape [B, T_text] or None
                num_tokens_original_all (torch.Tensor): Original token lengths for all samples in batch, shape [B]
                batch_size (int): Batch size
                device_target (torch.device): Target device to move tensors to
                
            Returns:
                tuple: (token_ids_b, token_len_b, text_token_ids_b, text_token_len_b) where:
                    - token_ids_b: Speech tokens for sample b_idx, shape [1, T_tok]
                    - token_len_b: Effective token length for sample b_idx (int)
                    - text_token_ids_b: Text tokens for sample b_idx, shape [1, T_text] or None
                    - text_token_len_b: Effective text length for sample b_idx (int)
            """
            token_ids_b = batch_dict['speech_token'][b_idx:b_idx+1].to(device_target)
            token_len_b = int(num_tokens_original_all[b_idx].item())
            
            # text tokens and effective length for this sample
            if isinstance(text_tokens_all, torch.Tensor) and text_tokens_all.size(0) == batch_size:
                text_token_ids_b = text_tokens_all[b_idx:b_idx+1].to(device_target)
            else:
                text_token_ids_b = text_tokens_all.to(device_target) if isinstance(text_tokens_all, torch.Tensor) else None
            
            text_token_lens = batch_dict.get('text_token_len', None)
            if isinstance(text_token_lens, torch.Tensor) and text_token_lens.numel() >= (b_idx + 1):
                text_token_len_b = int(text_token_lens[b_idx].item())
            else:
                text_token_len_b = int(text_token_ids_b.shape[1]) if isinstance(text_token_ids_b, torch.Tensor) else 0
            
            return token_ids_b, token_len_b, text_token_ids_b, text_token_len_b
        
        
        def _precompute_token_and_text_embeddings(token_ids_b, token_len_b, text_token_ids_b, text_token_len_b, upsample_f):
            """Precompute upsampled token embeddings and full text embeddings.
            
            Args:
                token_ids_b (torch.Tensor): Speech tokens for the current sample [1, T_tok]
                token_len_b (int): Effective token length for the current sample
                text_token_ids_b (torch.Tensor): Text tokens for the current sample [1, T_text] or None
                text_token_len_b (int): Effective text length for the current sample
                upsample_f (int): Upsampling factor used for token alignment
            
            Returns:
                tuple: (token_emb_full, num_tokens_upsampled, text_token_emb_full) where:
                    - token_emb_full: Upsampled token embeddings [1, num_tokens_upsampled, D]
                    - num_tokens_upsampled: Effective token length after upsampling
                    - text_token_emb_full: Full text embeddings [1, T_text, D] or None
            """
            # Upsample tokens
            if upsample_f > 1 and token_ids_b.numel() > 0:
                token_ids_upsampled = token_ids_b.repeat_interleave(upsample_f, dim=1)
            else:
                token_ids_upsampled = token_ids_b
            num_tokens_upsampled = token_ids_upsampled.shape[1]
            token_ids_upsampled = torch.clamp(token_ids_upsampled, min=0)
            token_emb_full = self.cos2_flow.input_embedding(token_ids_upsampled)
            
            # Pre-embed full text once
            if isinstance(text_token_ids_b, torch.Tensor) and text_token_ids_b.numel() > 0:
                text_token_ids_b = torch.clamp(text_token_ids_b[:, :text_token_len_b], min=0, max=self._text_vocab_size - 1)
                text_token_emb_full = self.text_context_emb(text_token_ids_b)
            else:
                text_token_emb_full = None
            
            return token_emb_full, num_tokens_upsampled, text_token_emb_full
        
        def _compute_first_chunk_length(block_size, token_len_b, upsample_f, b_idx, device_target):
            """Compute first chunk length (possibly randomized) and log if needed.
            
            Args:
                block_size (int): Chunking block size in original token units
                token_len_b (int): Effective token length for the current sample
                upsample_f (int): Upsampling factor used for token alignment
                b_idx (int): Batch index of the current sample
                device_target (torch.device): Target device for computations
                
            Returns:
                first_len_token (int): First chunk length in original token units
            """
            if self._stream_train_first_block_random and block_size > 0:
                first_len_token = int(torch.randint(low=1, high=block_size + 1, size=(1,), device=device_target).item())
            else:
                first_len_token = int(block_size)
            first_len_token = max(1, min(first_len_token, int(token_len_b))) if int(token_len_b) > 0 else 0
            first_len_upsampled = int(first_len_token * int(upsample_f)) if first_len_token > 0 else 0
            
            # training-time log: show randomized first block length (once per batch: b==0)
            if first_len_token > 0 and b_idx == 0 and (int(self._step) % max(self._print_per_n_chunk, 1) == 0):
                block_size_upsampled = max(1, block_size * int(upsample_f))
                num_tokens_upsampled_real = int(token_len_b * upsample_f)
                logging.info(f"[cos2.train.rand_first] first_len_token={first_len_token} first_len_upsampled={first_len_upsampled} block_size={block_size} block_size_upsampled={block_size_upsampled} num_tokens_upsampled_real={num_tokens_upsampled_real}")
            
            return first_len_token
        
        def _build_chunk_boundaries(token_len_b, first_len_token, block_size):
            """Build chunk start/end positions on original token axis.
            
            Args:
                token_len_b (int): Effective token length for the current sample
                first_len_token (int): First chunk length in original token units
                block_size (int): Chunking block size in original token units
            
            Returns:
                tuple: (chunk_starts_token, chunk_ends_token, num_chunks) where:
                    - chunk_starts_token: Start indices for each chunk on token axis
                    - chunk_ends_token: End indices for each chunk on token axis
                    - num_chunks: Number of chunks in the batch
            """
            if int(token_len_b) <= 0:
                return [], [], 0
            
            chunk_starts_token = [0]
            if first_len_token < int(token_len_b):
                chunk_starts_token += list(range(first_len_token, int(token_len_b), int(block_size)))
            
            # corresponding ends
            chunk_ends_token = []
            for i, st in enumerate(chunk_starts_token):
                if i == 0:
                    en = min(st + first_len_token, int(token_len_b))
                else:
                    en = min(st + int(block_size), int(token_len_b))
                chunk_ends_token.append(en)
            
            num_chunks = len(chunk_starts_token)
            return chunk_starts_token, chunk_ends_token, num_chunks
        
        def _slice_and_batch_token_embeddings(
            chunk_starts_token, chunk_ends_token, token_emb_full, upsample_f, token_len_b, 
            block_size, device_target
        ):
            """Slice token embeddings per chunk and build padded batch tensor.
            
            Args:
                chunk_starts_token (list): Start indices for each chunk on token axis
                chunk_ends_token (list): End indices for each chunk on token axis
                token_emb_full (torch.Tensor): Upsampled token embeddings [1, num_tokens_upsampled, D]
                upsample_f (int): Upsampling factor used for token alignment
                token_len_b (int): Effective token length for the current sample
                block_size (int): Chunking block size in original token units
                device_target (torch.device): Target device for computations

            Returns:
                tuple: (token_batch, len_vec, L_max) where:
                    - token_batch: Padded token embeddings [Nchunk, Lmax, D]
                    - len_vec: Sequence lengths [Nchunk]
                    - L_max: Maximum length of token embeddings in the batch
            """
            from cosyvoice.utils.mask import make_pad_mask
            
            block_size_upsampled = max(1, block_size * int(upsample_f))
            num_tokens_upsampled_real = int(token_len_b * upsample_f)
            
            token_slices = []
            token_lens = []
            for i, st_token in enumerate(chunk_starts_token):
                idx_upsampled = int(st_token) * int(upsample_f)
                end_token = int(chunk_ends_token[i])
                chunk_len_token = max(0, end_token - int(st_token))
                len_upsampled = int(max(0, min(int(chunk_len_token) * int(upsample_f), num_tokens_upsampled_real - idx_upsampled)))
                if len_upsampled <= 0:
                    continue
                te = token_emb_full[:, idx_upsampled: idx_upsampled + len_upsampled]
                if self._fixed_window_pad and te.shape[1] < block_size_upsampled:
                    pad = te.new_zeros(1, block_size_upsampled - te.shape[1], te.shape[2])
                    te = torch.cat([te, pad], dim=1)
                token_slices.append(te.squeeze(0))
                token_lens.append(len_upsampled)
            
            if len(token_lens) == 0:
                return None, None, None
            
            # pad to [Nchunk, Lmax, D]
            L_max = block_size_upsampled if self._fixed_window_pad else max(token_lens)
            D = token_emb_full.shape[-1]
            token_batch = token_emb_full.new_zeros((len(token_slices), L_max, D))
            for i, te in enumerate(token_slices):
                te_len = min(te.shape[0], L_max)
                token_batch[i, :te_len] = te[:te_len]
            
            len_vec = torch.tensor(token_lens, dtype=torch.int32, device=device_target)
            token_mask = (~make_pad_mask(len_vec, L_max)).float().unsqueeze(-1).to(device_target)
            token_batch = token_batch * token_mask
            
            return token_batch, len_vec, L_max
        
        def _build_text_context_for_chunks(
            text_token_emb_full, text_token_len_b, chunk_starts_token, chunk_ends_token, num_chunks, 
            token_ids_b, token_len_b, b_idx, device_target
        ):
            """Build batched text context for all chunks (debug + cross-attn or fallback).
            
            Args:
                text_token_emb_full (torch.Tensor): Full text embeddings [1, N, D]
                text_token_len_b (int): Text token length for the current sample
                chunk_starts_token (list): Start indices for each chunk on token axis
                chunk_ends_token (list): End indices for each chunk on token axis
                num_chunks (int): Number of chunks in the batch
                token_ids_b (torch.Tensor): Speech tokens for the current sample [1, T_tok]
                token_len_b (int): Effective token length for the current sample
                b_idx (int): Batch index of the current sample
                device_target (torch.device): Target device for computations

            Variables:
                L_ctx (int): Context length for text context (self._L_ctx_default)
                t_ends (list): End indices for each chunk on text axis
                kv_hist_max (int): Maximum visible text length for cross-attention

            Returns:
                text_ctx_batch (torch.Tensor): Text context batch [N, L_ctx, D] or None
            """
            if not isinstance(text_token_emb_full, torch.Tensor) or text_token_emb_full.numel() == 0:
                return None
            
            L_ctx = self._L_ctx_default
            t_ends = [int(min(int(e), int(text_token_len_b))) for e in chunk_ends_token]
            kv_hist_max = int(min(int(text_token_len_b), text_token_emb_full.shape[1]))
            
            # optional debug: show alignment for chunks spaced every K within the batch
            if getattr(self, '_debug_text_align', False) and (b_idx == 0) and ((int(self._step) % max(self._print_per_n_chunk, 1)) == 0):
                K = int(getattr(self, '_print_every_k_in_batch', 10))
                for i in range(num_chunks):
                    if (i % max(K,1) != 0) and (i != num_chunks - 1):
                        continue
                    try:
                        st_token_i = int(chunk_starts_token[i])
                        en_token_i = int(chunk_ends_token[i])
                        te = int(t_ends[i])
                        kvlen = int(min(te, kv_hist_max))
                        ctx_src = 'xattn' if self._use_cross_text_attn else 'prefix'
                        logging.info(f"[cos2.train.text-align] b={b_idx} chunk={i}/{num_chunks} t_range=[{st_token_i}:{en_token_i}) -> t_end={te}/{int(text_token_len_b)} kv_used=[0:{kvlen}) ctx={ctx_src} L_ctx={L_ctx} (token=[{st_token_i}:{en_token_i}))")
                    except Exception as e:
                        logging.info(f"[cos2.train.text-align] warn: {e}")
            
            # Build text context using cross-attention or fallback
            if self._use_cross_text_attn and kv_hist_max > 0:
                token_ids_original = torch.clamp(token_ids_b[:, :token_len_b], min=0)
                token_emb_original = self.cos2_flow.input_embedding(token_ids_original)
                text_ctx_batch = _build_cross_attention_text_context(
                    text_token_emb_full, chunk_starts_token, chunk_ends_token, token_len_b,
                    token_emb_original, num_chunks, kv_hist_max, t_ends, device_target, b_idx
                )
            elif not self._use_cross_text_attn:
                text_ctx_batch = _build_fallback_text_context(text_token_emb_full, t_ends, L_ctx, device_target)
            else:
                text_ctx_batch = None
            
            return text_ctx_batch
        
        def _reconstruct_sample_timeline(hidden_chunks, hidden_chunks_mask, num_chunks, device_target):
            """Reconstruct per-sample timeline by concatenating valid parts of each chunk.
            
            Args:
                hidden_chunks (torch.Tensor): Encoded hidden states from encoder [N, T_h, D]
                hidden_chunks_mask (torch.Tensor): Encoder output masks [N, 1, T_h] or similar
                num_chunks (int): Number of chunks in the batch
                device_target (torch.device): Target device for computations

            Returns:
                tuple: (hidden_b, mask_b, Li_list) where:
                    - hidden_b: Concatenated hidden states [1, T_h_total, D]
                    - mask_b: Concatenated masks [1, 1, T_h_total]
                    - Li_list: List of valid lengths for each chunk [N]
            """
            hidden_parts = []
            mask_parts = []
            Li_list = []
            
            for i in range(num_chunks):
                try:
                    hidden_chunks_mask_i = hidden_chunks_mask[i, 0]
                    if hidden_chunks_mask_i.dtype != torch.bool:
                        hidden_chunks_mask_i = hidden_chunks_mask_i > 0
                    Li = int(hidden_chunks_mask_i.sum().item())
                except Exception:
                    Li = int(hidden_chunks.shape[1])
                Li_list.append(Li)
                hidden_parts.append(hidden_chunks[i:i+1, :Li])
                
                if isinstance(hidden_chunks_mask, torch.Tensor):
                    hidden_chunks_mask_slice = hidden_chunks_mask[i:i+1, :, :Li]
                    if hidden_chunks_mask_slice.dtype != torch.bool:
                        hidden_chunks_mask_slice = hidden_chunks_mask_slice > 0
                    mask_parts.append(hidden_chunks_mask_slice)
                else:
                    mask_parts.append(torch.ones(1, 1, Li, dtype=torch.bool, device=device_target))
            
            if len(hidden_parts) > 0:
                hidden_b = torch.cat(hidden_parts, dim=1)
                hidden_b = self.cos2_flow.encoder_proj(hidden_b)
                try:
                    mask_b = torch.cat(mask_parts, dim=-1)
                except Exception:
                    mask_b = torch.ones(1, 1, hidden_b.shape[1], dtype=torch.bool, device=hidden_b.device)
            else:
                hidden_b = torch.zeros(1, 0, self.MEL_DIM, device=device_target)
                mask_b = torch.zeros(1, 1, 0, dtype=torch.bool, device=device_target)
            
            return hidden_b, mask_b, Li_list
        
        def _print_microbatch_debug_info(
            b_idx, num_chunks, L_max, upsample_f, block_size, kv_hist_max, 
            num_tokens_upsampled, Li_list, token_len_b, device_target
        ):
            """Print micro-batch debug statistics (first sample only, periodic).
            
            Args:
                b_idx (int): Batch index of the current sample
                num_chunks (int): Number of chunks in the batch
                L_max (int): Maximum length of token embeddings in the batch
                upsample_f (int): Upsampling factor used for token alignment
                block_size (int): Chunking block size in original token units
                kv_hist_max (int): Maximum visible text length for cross-attention
                num_tokens_upsampled (int): Effective token length after upsampling
                Li_list (list): List of valid lengths for each chunk [N]
                token_len_b (int): Effective token length for the current sample
                device_target (torch.device): Target device for computations
                
            Returns:
                None: This function only prints debug information to logs
            """
            if not (self._val_debug and b_idx == 0 and (int(self._step) % max(self._debug_every, 1) == 0)):
                return
            
            try:
                kv_hist_max_val = int(kv_hist_max) if 'kv_hist_max' in locals() else 0
            except Exception:
                kv_hist_max_val = 0
            
            L_ctx_val = int(self._L_ctx_default)
            
            try:
                mem_alloc = torch.cuda.memory_allocated(device_target) if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved(device_target) if torch.cuda.is_available() else 0
            except Exception:
                mem_alloc, mem_reserved = 0, 0
            
            try:
                tmr_dbg = float(getattr(self.cos2_flow, 'token_mel_ratio', 4.0))
            except Exception:
                tmr_dbg = 4.0
            
            try:
                sum_L = int(sum(Li_list)) if Li_list and len(Li_list) > 0 else 0
                Li_min = min(Li_list) if Li_list and len(Li_list) > 0 else 0
                Li_max = max(Li_list) if Li_list and len(Li_list) > 0 else 0
                Li_avg = (sum_L / max(len(Li_list), 1)) if Li_list and len(Li_list) > 0 else 0
            except Exception:
                sum_L, Li_min, Li_max, Li_avg = 0, 0, 0, 0
            
            exp_L = int(round(token_len_b * tmr_dbg))
            
            try:
                logging.info(
                    f"[cos2.train.mb] b={b_idx} chunks={num_chunks} Lmax={L_max} up={upsample_f} block={block_size} xattn={self._use_cross_text_attn} kv_hist_max={kv_hist_max_val} L_ctx={L_ctx_val} num_tokens_upsampled={num_tokens_upsampled} | Li(min/avg/max/sum)={Li_min}/{Li_avg:.1f}/{Li_max}/{sum_L} exp_L≈{exp_L} | mem(MB) alloc={mem_alloc/1e6:.1f} reserved={mem_reserved/1e6:.1f}"
                )
            except Exception:
                pass
        
        def _pad_and_concatenate_batch(hidden_list, mask_list):
            """Pad all samples to max time and concatenate into batch.
            
            Args:
                hidden_list (list): List of hidden states from encoder [N, T_h, D]
                mask_list (list): List of encoder output masks [N, 1, T_h] or similar
                
            Returns:
                tuple: (hidden, hidden_mask) where:
                    - hidden: Concatenated hidden states [B, T_h_total, D]
                    - hidden_mask: Concatenated masks [B, 1, T_h_total]
            """
            time_lengths = [hidden_i.shape[1] for hidden_i in hidden_list]
            max_time_length = max(time_lengths) if len(time_lengths) > 0 else 0
            H_cat = []
            M_cat = []
            for hidden_b, mask_b in zip(hidden_list, mask_list):
                if hidden_b.shape[1] < max_time_length:
                    pad_len = max_time_length - hidden_b.shape[1]
                    hidden_b = torch.cat([hidden_b, hidden_b.new_zeros(hidden_b.shape[0], pad_len, hidden_b.shape[2])], dim=1)
                    mask_b = torch.cat([mask_b, torch.zeros(mask_b.shape[0], mask_b.shape[1], pad_len, dtype=torch.bool, device=mask_b.device)], dim=-1)
                H_cat.append(hidden_b)
                M_cat.append(mask_b)
            hidden = torch.cat(H_cat, dim=0) if len(H_cat) > 1 else H_cat[0]
            hidden_mask = torch.cat(M_cat, dim=0) if len(M_cat) > 1 else M_cat[0]
            return hidden, hidden_mask
        
        # streaming_with_text_context prob from config
        streaming = torch.rand(()) < float(getattr(self, "_stream_train_prob", 0.5))
        
        # Infer actual device and prepare inputs
        device = _infer_device_from_batch(batch, device)
        token, token_len, feat, feat_len, embedding = _prepare_inputs_and_ensure_device(batch, device)

        # Keep original token ids for streaming-chunked path; compute upsample factor only
        num_tokens_original = token_len.clone()
        upsample_factor = self._compute_upsample_factor()

        # Optional: build text context for training (used differently for non-streaming vs streaming)
        text_tokens = batch.get('text_tokens', None) if isinstance(batch, dict) else None
        L_ctx = int(getattr(self.cos2_flow.encoder.pre_lookahead_layer, 'pre_lookahead_len', 4))

        from cosyvoice.utils.mask import make_pad_mask  # local import after sys.path patch

        if bool(streaming) and self._use_text_context_train:
            # Streaming-like training: per-sample, per-chunk encode with sliding text window
            batch_size = token.shape[0]
            hidden_list = []
            mask_list = []
            
            for batch_idx in range(batch_size):
                # Extract sample tokens and text
                token_ids_b, token_len_b, text_token_ids_b, text_token_len_b = _extract_sample_tokens_and_text(
                    batch, batch_idx, text_tokens, num_tokens_original, batch_size, device
                )
                
                # Determine chunk block size
                block_size = self._determine_chunk_block_size()
                
                # Precompute upsampled token and text embeddings
                token_emb_full, num_tokens_upsampled, text_token_emb_full = _precompute_token_and_text_embeddings(
                    token_ids_b, token_len_b, text_token_ids_b, text_token_len_b, upsample_factor
                )
                
                # Compute first chunk length (possibly randomized)
                first_len_token = _compute_first_chunk_length(block_size, token_len_b, upsample_factor, batch_idx, device)
                
                # Build chunk boundaries
                chunk_starts_token, chunk_ends_token, num_chunks = _build_chunk_boundaries(
                    token_len_b, first_len_token, block_size
                )
                
                # Handle empty chunks case
                if num_chunks == 0:
                    hidden_b = torch.zeros(1, 0, self.MEL_DIM, device=device)
                    mask_b = torch.zeros(1, 1, 0, dtype=torch.bool, device=device)
                    hidden_list.append(hidden_b)
                    mask_list.append(mask_b)
                    continue
                
                # Slice and batch token embeddings
                token_batch, len_vec, L_max = _slice_and_batch_token_embeddings(
                    chunk_starts_token, chunk_ends_token, token_emb_full, upsample_factor, 
                    token_len_b, block_size, device
                )
                
                # Handle case where slicing produced no valid chunks
                if token_batch is None:
                    hidden_b = torch.zeros(1, 0, self.MEL_DIM, device=device)
                    mask_b = torch.zeros(1, 1, 0, dtype=torch.bool, device=device)
                    hidden_list.append(hidden_b)
                    mask_list.append(mask_b)
                    continue
                
                # Build text context for all chunks
                text_ctx_batch = _build_text_context_for_chunks(
                    text_token_emb_full, text_token_len_b, chunk_starts_token, chunk_ends_token, num_chunks,
                    token_ids_b, token_len_b, batch_idx, device
                )
                
                # Encode chunks with text context
                if isinstance(text_ctx_batch, torch.Tensor) and text_ctx_batch.numel() > 0:
                    hidden_chunks, hidden_chunks_mask = self.cos2_flow.encoder(token_batch, len_vec, context=text_ctx_batch, streaming=True)
                else:
                    hidden_chunks, hidden_chunks_mask = self.cos2_flow.encoder(token_batch, len_vec, streaming=True)
                
                # Reconstruct per-sample timeline
                hidden_b, mask_b, Li_list = _reconstruct_sample_timeline(hidden_chunks, hidden_chunks_mask, num_chunks, device)
                
                hidden_list.append(hidden_b)
                mask_list.append(mask_b)
                
                # Print micro-batch debug info
                _print_microbatch_debug_info(
                    batch_idx, num_chunks, L_max, upsample_factor, block_size, 
                    text_token_emb_full.shape[1] if isinstance(text_token_emb_full, torch.Tensor) else 0,
                    num_tokens_upsampled, Li_list, token_len_b, device
                )
            
            # Pad and concatenate across batch
            hidden, hidden_mask = _pad_and_concatenate_batch(hidden_list, mask_list)
        else:
            # Non-streaming training: single pass with (global) text context prefix
            hidden, hidden_mask = _process_nonstreaming_path(token, token_len, text_tokens, upsample_factor, streaming, device)

        # Build partial cond prefix and compute decoder loss
        loss, lengths = _build_condition_and_compute_loss(hidden, hidden_mask, feat, feat_len, embedding, streaming)

        # Print periodic training-time debug information
        _print_training_debug_info(loss, streaming, token_len, num_tokens_original, hidden, lengths, upsample_factor, token)
        
        # advance internal step counter after prints
        self._step += 1
        return {'loss': loss}

    @torch.inference_mode()
    def token2wav(
        self,
        token: torch.Tensor,
        uuid: str,
        prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int64),
        prompt_feat: torch.Tensor = None,
        embedding: torch.Tensor = None,
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
            tts_speech_22k: [B, T_wav] waveform at 22050 Hz for compatibility
            tts_mel: [B, 80, T_mel] generated mel
        """
        # Infer device from inputs to avoid cross-device ops under DDP
        device = token.device

        # Ensure cos2_flow on correct device for inference
        self._ensure_device(device)

        # Lazy init HiFT on first use (move to correct device if needed)
        self._lazy_init_hift()
        hift_dev = next(self._hift.parameters()).device
        if hift_dev != device:
            self._hift.to(device)

        batch_size = token.shape[0]

        # CosyVoice2 flow.inference 目前断言 batch=1，这里逐样本推理再拼回 batch。
        tts_mels = []
        vocab_max = None  # WhisperVQ 16384; keep None unless you need clamp
        # determine pre-upsample factor to align token->encoder time
        upsample_factor = self._compute_upsample_factor()

        for batch_idx in range(batch_size):
            token_b = torch.clamp(token[batch_idx : batch_idx + 1].to(device), min=0)
            ptok_b = prompt_token[batch_idx : batch_idx + 1].to(device) if (prompt_token is not None and prompt_token.numel() > 0 and prompt_token.size(0) == batch_size) else (prompt_token.to(device) if (prompt_token is not None and prompt_token.numel() > 0) else torch.zeros(1, 0, dtype=torch.int64, device=device))
            pfeat_b = prompt_feat[batch_idx : batch_idx + 1].to(device) if (prompt_feat is not None and prompt_feat.numel() > 0 and prompt_feat.size(0) == batch_size) else (prompt_feat.to(device) if (prompt_feat is not None and prompt_feat.numel() > 0) else torch.zeros(1, 0, self.MEL_DIM, device=device))
            emb_b = (embedding[batch_idx : batch_idx + 1].to(device) if (embedding is not None and embedding.size(0) == batch_size) else (embedding.to(device) if embedding is not None else torch.zeros(1, self.SPEAKER_EMBEDDING_DIM, device=device)))

            if vocab_max is not None and vocab_max >= 0:
                token_b = torch.clamp(token_b, min=0, max=vocab_max)
                if ptok_b.numel() > 0:
                    ptok_b = torch.clamp(ptok_b, min=0, max=vocab_max)

            # Avoid NaNs from F.normalize on zero vector inside CosyVoice2 flow
            if torch.all(emb_b == 0):
                emb_b = emb_b.clone()
                emb_b[:, 0] = 1e-6

            # Pre-upsample tokens so that encoder x2 matches token_mel_ratio (~4 -> x2)
            if upsample_factor > 1:
                token_upsampled = token_b.repeat_interleave(upsample_factor, dim=1)
                ptok_upsampled = ptok_b.repeat_interleave(upsample_factor, dim=1) if (ptok_b is not None and ptok_b.numel() > 0) else ptok_b
                token_len_upsampled = token_upsampled.shape[1]
                ptok_len_upsampled = ptok_upsampled.shape[1] if (ptok_upsampled is not None and ptok_upsampled.numel() > 0) else 0
            else:
                token_upsampled = token_b
                ptok_upsampled = ptok_b
                token_len_upsampled = token_upsampled.shape[1]
                ptok_len_upsampled = ptok_upsampled.shape[1] if (ptok_upsampled is not None and ptok_upsampled.numel() > 0) else 0

            mel_b, _ = self.cos2_flow.inference(
                token=token_upsampled,
                token_len=torch.tensor([token_len_upsampled], dtype=torch.int32, device=device),
                prompt_token=ptok_upsampled,
                prompt_token_len=torch.tensor([ptok_len_upsampled], dtype=torch.int32, device=device),
                prompt_feat=pfeat_b,
                prompt_feat_len=torch.tensor([pfeat_b.shape[1]], dtype=torch.int32, device=device),
                embedding=emb_b,
                streaming=False,
                finalize=True,
            )
            tts_mels.append(mel_b)
        # [B, 80, T]
        tts_mel = torch.cat(tts_mels, dim=0)

        # Optional debug prints during validation only
        if self._val_debug:
            with torch.no_grad():
                logging.info(f"[cos2.val] device={device} B={batch_size} token_min={token.min().item()} token_max={token.max().item()} ")
                logging.info(f"[cos2.val] mel shape={tts_mel.shape} mel mean/std={tts_mel.mean().item():.4f}/{tts_mel.std().item():.4f}")

        # HiFT vocoder: mel->wav (CosyVoice2 uses 24kHz by default)
        tts_speech_24k, _ = self._hift.inference(speech_feat=tts_mel)
        # Keep downstream unchanged: resample 24k -> 22.05k to match existing code paths
        out_sr = self.OUTPUT_SAMPLE_RATE
        if self._cos2_sr != out_sr:
            tts_speech = nemo_resample(tts_speech_24k, self._cos2_sr, out_sr)
        else:
            tts_speech = tts_speech_24k

        if self._val_debug:
            try:
                logging.info(f"[cos2.val] wav22050 shape={tts_speech.shape} mean/std={tts_speech.mean().item():.4f}/{tts_speech.std().item():.4f}")
            except Exception:
                pass

        return tts_speech, tts_mel

    @torch.inference_mode()
    def stream_inference(
        self,
        token: torch.Tensor,
        uuid: str,
        prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int64),
        prompt_feat: torch.Tensor = torch.zeros(1, 0, 80),
        embedding: torch.Tensor = torch.zeros(1, 192),
        gt_mel_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Streaming decode tokens to waveform using CosyVoice2 flow + HiFT with overlap-fade.

        Args:
            token: [B, T_tok]
            uuid: stream session id
            prompt_token: [B, Tp]
            prompt_feat: [B, Tp_mel, 80]
            embedding: [B, 192]
        Returns:
            wav22050: [B, T_wav] at 22050 Hz
        """
        device = token.device
        # resolve block_size/token_fps robustly
        block_size = self._determine_chunk_block_size()
        try:
            token_fps = float(getattr(self.cos2_flow, 'input_frame_rate', 25.0))
        except Exception:
            token_fps = 25.0

        # Ensure modules on correct device
        self._ensure_device(device)
        self._lazy_init_hift()
        try:
            hift_dev = next(self._hift.parameters()).device
        except StopIteration:
            hift_dev = device
        if hift_dev != device:
            self._hift.to(device)

        batch_size = token.shape[0]
        # If batch>1, process each sample independently (flow streaming is batch-1 safe)
        if batch_size > 1:
            wav_list = []
            max_len = 0
            for batch_idx in range(batch_size):
                uid_b = f"{uuid}#{batch_idx}"
                token_b = token[batch_idx : batch_idx + 1]
                ptok_b = prompt_token[batch_idx : batch_idx + 1] if prompt_token is not None and hasattr(prompt_token, 'size') and prompt_token.size(0) == batch_size else prompt_token
                pfeat_b = prompt_feat[batch_idx : batch_idx + 1] if prompt_feat is not None and hasattr(prompt_feat, 'size') and prompt_feat.size(0) == batch_size else prompt_feat
                emb_b = embedding[batch_idx : batch_idx + 1] if embedding is not None and hasattr(embedding, 'size') and embedding.size(0) == batch_size else embedding
                gt_b = gt_mel_len[batch_idx : batch_idx + 1] if (gt_mel_len is not None and isinstance(gt_mel_len, torch.Tensor) and gt_mel_len.size(0) == batch_size) else gt_mel_len
                wav_b = self.stream_inference(token_b, uid_b, ptok_b, pfeat_b, emb_b, gt_b)
                wav_list.append(wav_b)
                max_len = max(max_len, wav_b.shape[1])
            # right-pad to same length and stack
            padded = []
            for w in wav_list:
                if w.shape[1] < max_len:
                    pad = torch.zeros(w.shape[0], max_len - w.shape[1], device=w.device, dtype=w.dtype)
                    w = torch.cat([w, pad], dim=1)
                padded.append(w)
            return torch.cat(padded, dim=0)

        wav_chunks = []
        prev_mel = self._mel_overlap_dict[uuid]
        hift_cache = self._hift_cache_dict[uuid]
        if hift_cache is not None:
            cache_src = hift_cache['source']
        else:
            cache_src = torch.zeros(batch_size, 1, 0, device=device)
        # reset total mel frame counter for this streaming session (start from prompt_feat length if provided)
        try:
            init_total = int(prompt_feat.shape[1]) if (prompt_feat is not None and prompt_feat.numel() > 0) else 0
        except Exception:
            init_total = 0
        self._mel_total_len_dict[uuid] = init_total
        self._mel_model_total_dict[uuid] = 0


        # iterate with sliding window: step by stream_stride tokens; window length = block_size
        stride = int(self._stream_stride) if getattr(self, "_stream_stride", 0) and int(self._stream_stride) > 0 else block_size
        T = token.size(1)
        step_i = 0
        for end in range(min(stride, T), T + 1, stride):
            start = max(0, end - block_size)
            # current window tokens [start:end]; no left-pad is passed to flow to keep token_len consistent
            token_win = token[:, start:end]
            real_len = token_win.shape[1]
            ptok = token[:, :start] if start > 0 else prompt_token
            pfeat = torch.cat([prompt_feat, prev_mel.transpose(1, 2)], dim=1) if (prev_mel is not None and prev_mel.numel() > 0) else prompt_feat

            # pre-upsample so that encoder x2 matches token_mel_ratio (~4 -> upsample_factor~2)
            upsample_factor = self._compute_upsample_factor()
            if upsample_factor > 1:
                token_upsampled = token_win.repeat_interleave(upsample_factor, dim=1)
                ptok_upsampled = ptok.repeat_interleave(upsample_factor, dim=1) if (ptok is not None and ptok.numel() > 0) else ptok
                real_len_upsampled = real_len * upsample_factor
                ptok_len_upsampled = (ptok.shape[1] * upsample_factor) if (ptok is not None and ptok.numel() > 0) else 0
            else:
                token_upsampled = token_win
                ptok_upsampled = ptok
                real_len_upsampled = real_len
                ptok_len_upsampled = ptok.shape[1] if (ptok is not None and ptok.numel() > 0) else 0

            # optional fixed window padding (right-pad ids); token_len stays real_len_upsampled
            if self._fixed_window_pad:
                block_size_upsampled = int(block_size * upsample_factor)
                if token_upsampled.shape[1] < block_size_upsampled:
                    pad_len = block_size_upsampled - token_upsampled.shape[1]
                    pad_ids = token_upsampled[:, -1:].expand(-1, pad_len)
                    token_upsampled = torch.cat([token_upsampled, pad_ids], dim=1)


            finalize = end >= T
            # sparse prints per N chunks
            if (step_i % max(self._print_per_n_chunk, 1)) == 0:
                logging.info(f"[cos2.stream] step={step_i} win=({start},{end}) stride={stride} token_real={real_len} up={upsample_factor} token_upsampled={token_upsampled.shape[1]} finalize={finalize}")

            # 1) generate mel for this window via CosyVoice2 flow (streaming=True)
            mel_b, _ = self.cos2_flow.inference(
                token=token_upsampled,
                token_len=torch.tensor([real_len_upsampled], dtype=torch.int32, device=device),
                prompt_token=ptok_upsampled,
                prompt_token_len=torch.tensor([ptok_len_upsampled], dtype=torch.int32, device=device),
                prompt_feat=pfeat,
                prompt_feat_len=torch.tensor([pfeat.shape[1]], dtype=torch.int32, device=device),
                embedding=embedding,
                streaming=True,
                finalize=finalize,
            )
            if (step_i % max(self._print_per_n_chunk, 1)) == 0:
                logging.info(f"[cos2.stream] mel_b_frames={mel_b.shape[-1]}")
            step_i += 1
            # mel_b: [B, 80, T_mel_new]

            # Model returns per-step cumulative frames w.r.t. prompt_token (not prompt_feat).
            # We must take the delta vs previous model cumulative to avoid duplication.
            T_all = int(mel_b.shape[-1])
            prev_session = int(self._mel_model_total_dict.get(uuid, 0))
            if T_all <= prev_session:
                if (step_i % max(self._print_per_n_chunk, 1)) == 0:
                    logging.info(f"[cos2.stream] uuid={uuid} T_all={T_all} prev_session={prev_session} -> delta=0 (skip)")
                continue
            start = prev_session
            delta = T_all - prev_session
            self._mel_model_total_dict[uuid] = T_all
            emitted_prev = int(self._mel_total_len_dict.get(uuid, 0))
            self._mel_total_len_dict[uuid] = emitted_prev + delta
            if (step_i % max(self._print_per_n_chunk, 1)) == 0:
                logging.info(f"[cos2.stream] uuid={uuid} start={start} delta={delta} emitted_total(prev)={emitted_prev} emitted_total(now)={self._mel_total_len_dict[uuid]}")
            new_mel = mel_b[:, :, start:T_all]

            # overlap-and-add on new frames and keep tail overlap for next chunk
            if not finalize and self._mel_overlap_len > 0:
                ol = int(self._mel_overlap_len)
                prev_len = int(prev_mel.shape[-1]) if (prev_mel is not None and prev_mel.numel() > 0) else 0
                new_len = int(new_mel.shape[-1])
                overlap_effective = min(ol, prev_len, new_len)
                if overlap_effective > 0:
                    # use the first overlap_effective weights from the first half and the first overlap_effective from the second half
                    w = torch.tensor(self._mel_window, device=new_mel.device, dtype=new_mel.dtype)
                    w1 = w[:overlap_effective].view(1, 1, overlap_effective)
                    w2 = w[ol:ol+overlap_effective].view(1, 1, overlap_effective)
                    new_mel[:, :, :overlap_effective] = new_mel[:, :, :overlap_effective] * w1 + prev_mel[:, :, -overlap_effective:] * w2
                # keep last min(ol, new_len) frames for next iteration
                keep = min(ol, new_len)
                self._mel_overlap_dict[uuid] = new_mel[:, :, -keep:]
                prev_mel = self._mel_overlap_dict[uuid]
                mel_for_vocoder = new_mel
            else:
                mel_for_vocoder = new_mel
            if (step_i % max(self._print_per_n_chunk, 1)) == 0:
                logging.info(f"[cos2.stream] mel_for_vocoder_frames={mel_for_vocoder.shape[-1]} ol={self._mel_overlap_len} total_mel={self._mel_total_len_dict[uuid]}")


            # clear caches on finalize
            if finalize:
                total = int(self._mel_total_len_dict.get(uuid, 0))
                # Build GT mel info and seconds if provided (GT mel is 22050Hz/hop256)
                gt_info = ""
                try:
                    if gt_mel_len is not None:
                        if isinstance(gt_mel_len, torch.Tensor):
                            if gt_mel_len.numel() == 1:
                                _v_list = [int(gt_mel_len.view(-1)[0].item())]
                            else:
                                _v_list = [int(x) for x in gt_mel_len.view(-1).tolist()]
                        else:
                            _v_list = [int(gt_mel_len)]
                        gt_sec_list = [round(v * 256.0 / float(self.OUTPUT_SAMPLE_RATE), 3) for v in _v_list]
                        _v = _v_list[0] if len(_v_list) == 1 else _v_list
                        _s = gt_sec_list[0] if len(gt_sec_list) == 1 else gt_sec_list
                        gt_info = f" gt_mel_len={_v} gt_sec={_s}"
                except Exception:
                    gt_info = " gt_mel_len=NA"
                # Emitted seconds at 24k/hop480 (=50 fps)
                emitted_sec = round(total * (self._mel_hop / float(self._cos2_sr)), 3)
                try:
                    e = embedding.detach().float() if isinstance(embedding, torch.Tensor) else None
                    if e is not None:
                        l2 = torch.norm(e, dim=1).mean().item() if e.ndim == 2 and e.size(0) > 0 else float(torch.norm(e).item())
                        head = e[0, :8].tolist() if e.ndim == 2 and e.size(0) > 0 else []
                        logging.info(f"[cos2.stream] uuid={uuid} finalize=True emitted_total={total} emitted_sec={emitted_sec}{gt_info} | spk shape={list(e.shape)} mean={e.mean().item():.5f} std={e.std().item():.5f} l2_mean={l2:.5f} head8={head}")
                    else:
                        logging.info(f"[cos2.stream] uuid={uuid} finalize=True emitted_total={total} emitted_sec={emitted_sec}{gt_info} | spk=NA")
                except Exception:
                    logging.info(f"[cos2.stream] uuid={uuid} finalize=True emitted_total={total} emitted_sec={emitted_sec}{gt_info} | spk=ERR")
                self._mel_overlap_dict.pop(uuid, None)
                self._hift_cache_dict.pop(uuid, None)
                self._mel_total_len_dict.pop(uuid, None)


            # 3) HiFT vocoder with cache to avoid glitch
            speech_24k, source = self._hift.inference(speech_feat=mel_for_vocoder, cache_source=cache_src)

            # update source cache
            if not finalize:
                cache_src = source[:, :, -self._source_cache_len:]
                self._hift_cache_dict[uuid] = {
                    'source': cache_src,
                }
                # Do not drop tail samples; HiFT cache ensures continuity without duplication.
            else:
                # clear cache
                cache_src = torch.zeros(batch_size, 1, 0, device=device)

            # Accumulate 24k chunks; resample once at the end to avoid boundary artifacts
            wav_chunks.append(speech_24k)


        # Concatenate all 24k chunks and resample once at the end (baseline non-overlap)
        if len(wav_chunks) == 0:
            return torch.zeros(batch_size, 0, device=device)
        wav_24k = torch.cat(wav_chunks, dim=-1)
        out_sr = self.OUTPUT_SAMPLE_RATE
        if self._cos2_sr != out_sr:
            wav = nemo_resample(wav_24k, self._cos2_sr, out_sr)
        else:
            wav = wav_24k
        return wav


    @torch.inference_mode()
    def stream_inference_with_text(
        self,
        token: torch.Tensor,
        uuid: str,
        text_tokens: Optional[torch.Tensor] = None,
        prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int64),
        prompt_feat: torch.Tensor = torch.zeros(1, 0, 80),
        embedding: torch.Tensor = torch.zeros(1, 192),
        gt_mel_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Streaming decode with text-window context replacing semantic pre-lookahead.

        Args:
            token: [B, T_tok]
            uuid: session id
            text_tokens: [B, T_text] text token ids (LLM head outputs). If None, fallback to vanilla stream_inference.
            prompt_token: [B, Tp]
            prompt_feat: [B, Tp_mel, 80]
            embedding: [B, 192]
        Returns:
            wav22050: [B, T_wav]
        """
        if text_tokens is None or text_tokens.numel() == 0:
            return self.stream_inference(token, uuid, prompt_token, prompt_feat, embedding)

        device = token.device
        # resolve block_size/token_fps robustly
        block_size = self._determine_chunk_block_size()
        try:
            token_fps = float(getattr(self.cos2_flow, 'input_frame_rate', 25.0))
        except Exception:
            token_fps = 25.0

        # Ensure modules on correct device
        self._ensure_device(device)
        self._lazy_init_hift()
        try:
            hift_dev = next(self._hift.parameters()).device
        except StopIteration:
            hift_dev = device
        if hift_dev != device:
            self._hift.to(device)

        batch_size = token.shape[0]
        if batch_size > 1:
            wav_list = []
            max_len = 0
            for batch_idx in range(batch_size):
                uid_b = f"{uuid}#{batch_idx}"
                token_b = token[batch_idx : batch_idx + 1]
                ttxt_b = text_tokens[batch_idx : batch_idx + 1] if text_tokens is not None and hasattr(text_tokens, 'size') and text_tokens.size(0) == batch_size else text_tokens
                ptok_b = prompt_token[batch_idx : batch_idx + 1] if prompt_token is not None and hasattr(prompt_token, 'size') and prompt_token.size(0) == batch_size else prompt_token
                pfeat_b = prompt_feat[batch_idx : batch_idx + 1] if prompt_feat is not None and hasattr(prompt_feat, 'size') and prompt_feat.size(0) == batch_size else prompt_feat
                emb_b = embedding[batch_idx : batch_idx + 1] if embedding is not None and hasattr(embedding, 'size') and embedding.size(0) == batch_size else embedding
                gt_b = gt_mel_len[batch_idx : batch_idx + 1] if (gt_mel_len is not None and isinstance(gt_mel_len, torch.Tensor) and gt_mel_len.size(0) == batch_size) else gt_mel_len
                wav_b = self.stream_inference_with_text(token_b, uid_b, ttxt_b, ptok_b, pfeat_b, emb_b, gt_b)
                wav_list.append(wav_b)
                max_len = max(max_len, wav_b.shape[1])
            # right-pad to same length and stack
            padded = []
            for w in wav_list:
                if w.shape[1] < max_len:
                    pad = torch.zeros(w.shape[0], max_len - w.shape[1], device=w.device, dtype=w.dtype)
                    w = torch.cat([w, pad], dim=1)
                padded.append(w)
            return torch.cat(padded, dim=0)

        wav_chunks = []
        prev_mel = self._mel_overlap_dict[uuid]
        hift_cache = self._hift_cache_dict[uuid]
        if hift_cache is not None:
            cache_src = hift_cache['source']
        else:
            cache_src = torch.zeros(batch_size, 1, 0, device=device)
        # reset total mel frame counter for this streaming session (start from prompt_feat length if provided)
        try:
            init_total = int(prompt_feat.shape[1]) if (prompt_feat is not None and prompt_feat.numel() > 0) else 0
        except Exception:
            init_total = 0
        self._mel_total_len_dict[uuid] = init_total
        self._mel_model_total_dict[uuid] = 0

        # iterate with sliding window + text context
        stride = int(self._stream_stride) if getattr(self, "_stream_stride", 0) and int(self._stream_stride) > 0 else block_size
        T = token.size(1)
        step_i = 0
        for end in range(min(stride, T), T + 1, stride):
            start = max(0, end - block_size)
            token_win = token[:, start:end]
            real_len = token_win.shape[1]

            ptok = token[:, :start] if start > 0 else prompt_token
            pfeat = torch.cat([prompt_feat, prev_mel.transpose(1, 2)], dim=1) if (prev_mel is not None and prev_mel.numel() > 0) else prompt_feat

            # Cross-text attention (if enabled) else fallback to prefix window
            # Align text visibility to current speech token progress: t_end = min(T_text, end)
            t_end = min(int(text_tokens.shape[1]), int(end))
            if self._use_cross_text_attn:
                # KV from text up to t_end, with per-batch mask
                kv = self.text_context_emb(torch.clamp(text_tokens[:, :t_end], min=0, max=self._text_vocab_size - 1))  # [B, Lt, D]
                mask_kv = torch.ones(kv.shape[0], 1, kv.shape[1], dtype=torch.bool, device=kv.device)
                # Q from speech histories: use all past + current chunk tokens as self-attn context, then slice current chunk as multi-query
                token_hist = token[:, :end]
                token_hist_emb = self.cos2_flow.input_embedding(token_hist)  # [B, Lhist, D]
                Lhist = token_hist_emb.shape[1]
                tril = torch.tril(torch.ones((Lhist, Lhist), dtype=torch.bool, device=token_hist_emb.device))
                mask_s_hist = tril.unsqueeze(0).expand(token_hist_emb.shape[0], -1, -1)
                s_hist_attn, _ = self.speech_sa(query=token_hist_emb, key=token_hist_emb, value=token_hist_emb, mask=mask_s_hist)
                s_hist = self.speech_ln(s_hist_attn)
                s_hist = s_hist + self.speech_ffn(s_hist)
                # extract current chunk subrange [end-real_len:end]
                start_q = max(0, end - real_len)
                s = s_hist[:, start_q:end, :]  # [B, Lq, D]
                # Text refinement (self-attn+FFN)
                # Causal self-attn on text tokens up to t_end
                Lt = kv.shape[1]
                tril = torch.tril(torch.ones((Lt, Lt), dtype=torch.bool, device=kv.device))
                mask_kv_causal = tril.unsqueeze(0).expand(kv.shape[0], -1, -1)
                kv_attn, _ = self.text_sa(query=kv, key=kv, value=kv, mask=mask_kv_causal)
                kv_ref = self.text_ln(kv_attn)
                kv_ref = kv_ref + self.text_ffn(kv_ref)
                # Cross-attn with multi-query; then pool last valid step to single vector per batch
                attn_out, attn_w = self.cross_text_attn(query=s, key=kv_ref, value=kv_ref, mask=mask_kv)  # [B, Lq, D]
                if getattr(self, '_debug_xattn', False) and ((step_i % max(self._print_per_n_chunk, 1)) == 0):
                    try:
                        # attn_w expected shape [B,H,Q,K] or [B,Q,K]
                        if isinstance(attn_w, torch.Tensor):
                            if attn_w.dim() == 4:
                                aw = attn_w.mean(dim=1)
                            elif attn_w.dim() == 3:
                                aw = attn_w
                            else:
                                aw = None
                        else:
                            aw = None
                        Bq = int(s.shape[1]); Kt = int(kv_ref.shape[1])
                        if aw is not None and real_len > 0:
                            vec = aw[0, real_len - 1, :Kt]
                            k = int(min(3, Kt))
                            vec = torch.softmax(vec, dim=-1)
                            vals, idxs = torch.topk(vec, k)
                            logging.info(f"[cos2.infer.xattn] step={step_i} Q_len={Bq} K_len={Kt} topk_idx={idxs.tolist()} topk_val={[round(float(v),4) for v in vals.tolist()]}")
                        else:
                            logging.info(f"[cos2.infer.xattn] step={step_i} Q_len={Bq} K_len={Kt} (no attn_w)")
                    except Exception as e:
                        logging.info(f"[cos2.infer.xattn] warn: {e}")
                # Aggregate full-Q features to L_ctx via learned queries (no temporal pooling)
                q_ctx = self.ctx_queries.to(attn_out.device).unsqueeze(0).expand(attn_out.shape[0], -1, -1)  # [B,L_ctx,D]
                key_mask = torch.ones(attn_out.shape[0], 1, attn_out.shape[1], dtype=torch.bool, device=attn_out.device)
                ctx_raw, _ = self.q2ctx_attn(query=q_ctx, key=attn_out, value=attn_out, mask=key_mask)  # [B,L_ctx,D]
                x = self.cross_ln(ctx_raw)
                x = x + self.cross_ffn(x)
                L_ctx = int(getattr(self.cos2_flow.encoder.pre_lookahead_layer, 'pre_lookahead_len', 4))
                text_ctx = x  # already [B, L_ctx, D]
            else:
                emb = self.text_context_emb(torch.clamp(text_tokens[:, :t_end], min=0, max=self._text_vocab_size - 1))
                L_ctx = int(getattr(self.cos2_flow.encoder.pre_lookahead_layer, 'pre_lookahead_len', 4))
                if emb.shape[1] < L_ctx:
                    pad = torch.zeros(B, L_ctx - emb.shape[1], emb.shape[2], device=emb.device, dtype=emb.dtype)
                    emb_ctx = torch.cat([emb, pad], dim=1)
                else:
                    emb_ctx = emb[:, -L_ctx:]
                text_ctx = emb_ctx

            # upsample
            upsample_factor = self._compute_upsample_factor()
            if upsample_factor > 1:
                token_upsampled = token_win.repeat_interleave(upsample_factor, dim=1)
                ptok_upsampled = ptok.repeat_interleave(upsample_factor, dim=1) if (ptok is not None and ptok.numel() > 0) else ptok
                real_len_upsampled = real_len * upsample_factor
                ptok_len_upsampled = (ptok.shape[1] * upsample_factor) if (ptok is not None and ptok.numel() > 0) else 0
            else:
                token_upsampled = token_win
                ptok_upsampled = ptok
                real_len_upsampled = real_len
                ptok_len_upsampled = ptok.shape[1] if (ptok is not None and ptok.numel() > 0) else 0

            # optional fixed window padding (right-pad ids) for text-conditioned streaming
            if self._fixed_window_pad:
                block_size_upsampled = int(block_size * upsample_factor)
                if token_upsampled.shape[1] < block_size_upsampled:
                    pad_len = block_size_upsampled - token_upsampled.shape[1]
                    pad_ids = token_upsampled[:, -1:].expand(-1, pad_len)
                    token_upsampled = torch.cat([token_upsampled, pad_ids], dim=1)

            finalize = end >= T
            if (step_i % max(self._print_per_n_chunk, 1)) == 0:
                ctx_src = 'xattn' if self._use_cross_text_attn else 'prefix'
                try:
                    if self._fixed_window_pad:
                        pad_tok = max(0, int(block_size - real_len))
                        pad_upsampled = max(0, int(block_size * upsample_factor - real_len_upsampled))
                    else:
                        pad_tok = 0
                        pad_upsampled = 0
                except Exception:
                    pad_tok = 0
                    pad_upsampled = 0
                logging.info(
                    f"[cos2.stream+text] step={step_i} win=({start},{end}) stride={stride} token_real={real_len} up={upsample_factor} token_upsampled={token_upsampled.shape[1]} "
                    f"pad_tok={pad_tok} pad_upsampled={pad_upsampled} t_end={t_end} text=[0:{t_end}) q_pool=multi_query:ctx_attn ctx={ctx_src} L_ctx={L_ctx} finalize={finalize}"
                )



            # Run CosyVoice2 flow with text context replacing semantic lookahead
            mel_b, _ = self.cos2_flow.inference(
                token=token_upsampled,
                token_len=torch.tensor([real_len_upsampled], dtype=torch.int32, device=device),
                prompt_token=ptok_upsampled,
                prompt_token_len=torch.tensor([ptok_len_upsampled], dtype=torch.int32, device=device),
                prompt_feat=pfeat,
                prompt_feat_len=torch.tensor([pfeat.shape[1]], dtype=torch.int32, device=device),
                embedding=embedding,
                streaming=True,
                finalize=finalize,
                use_text_context=True,
                text_context=text_ctx,
            )
            # step index for logging
            step_i += 1

            # Same as non-text streaming: inference() returns cumulative frames w.r.t. prompt_token.
            # Compute delta vs previous model cumulative frames.
            T_all = int(mel_b.shape[-1])
            prev_session = int(self._mel_model_total_dict.get(uuid, 0))
            if T_all <= prev_session:
                # try:
                #     logging.info(f"[cos2.stream+text] uuid={uuid} T_all={T_all} prev_session={prev_session} -> delta=0 (skip)")
                # except Exception:
                #     pass
                continue
            start = prev_session
            delta = T_all - prev_session
            self._mel_model_total_dict[uuid] = T_all
            emitted_prev = int(self._mel_total_len_dict.get(uuid, 0))
            self._mel_total_len_dict[uuid] = emitted_prev + delta
            # try:
            #     logging.info(f"[cos2.stream+text] uuid={uuid} start={start} delta={delta} emitted_total(prev)={emitted_prev} emitted_total(now)={self._mel_total_len_dict[uuid]}")
            # except Exception:
            #     pass
            new_mel = mel_b[:, :, start:T_all]

            # overlap-and-add as usual
            if not finalize and self._mel_overlap_len > 0:
                ol = int(self._mel_overlap_len)
                prev_len = int(prev_mel.shape[-1]) if (prev_mel is not None and prev_mel.numel() > 0) else 0
                new_len = int(new_mel.shape[-1])
                overlap_effective = min(ol, prev_len, new_len)
                if overlap_effective > 0:
                    w = torch.tensor(self._mel_window, device=new_mel.device, dtype=new_mel.dtype)
                    w1 = w[:overlap_effective].view(1, 1, overlap_effective)
                    w2 = w[ol:ol+overlap_effective].view(1, 1, overlap_effective)
                    new_mel[:, :, :overlap_effective] = new_mel[:, :, :overlap_effective] * w1 + prev_mel[:, :, -overlap_effective:] * w2
                keep = min(ol, new_len)
                self._mel_overlap_dict[uuid] = new_mel[:, :, -keep:]
                prev_mel = self._mel_overlap_dict[uuid]
                mel_for_vocoder = new_mel
            else:
                mel_for_vocoder = new_mel

            # clear caches on finalize
            if finalize:
                total = int(self._mel_total_len_dict.get(uuid, 0))
                try:
                    e = embedding.detach().float() if isinstance(embedding, torch.Tensor) else None
                    if e is not None:
                        l2 = torch.norm(e, dim=1).mean().item() if e.ndim == 2 and e.size(0) > 0 else float(torch.norm(e).item())
                        head = e[0, :8].tolist() if e.ndim == 2 and e.size(0) > 0 else []
                        # Build GT mel info and seconds if provided (GT mel is 22050Hz/hop256)
                        gt_info = ""
                        try:
                            if gt_mel_len is not None:
                                if isinstance(gt_mel_len, torch.Tensor):
                                    if gt_mel_len.numel() == 1:
                                        _v_list = [int(gt_mel_len.view(-1)[0].item())]
                                    else:
                                        _v_list = [int(x) for x in gt_mel_len.view(-1).tolist()]
                                else:
                                    _v_list = [int(gt_mel_len)]
                                gt_sec_list = [round(v * 256.0 / float(self.OUTPUT_SAMPLE_RATE), 3) for v in _v_list]
                                _v = _v_list[0] if len(_v_list) == 1 else _v_list
                                _s = gt_sec_list[0] if len(gt_sec_list) == 1 else gt_sec_list
                                gt_info = f" gt_mel_len={_v} gt_sec={_s}"
                        except Exception:
                            gt_info = " gt_mel_len=NA"
                        # Emitted seconds at 24k/hop480 (=50 fps)
                        emitted_sec = round(total * (self._mel_hop / float(self._cos2_sr)), 3)
                        logging.info(f"[cos2.stream+text] uuid={uuid} finalize=True emitted_total={total} emitted_sec={emitted_sec}{gt_info} | spk shape={list(e.shape)} mean={e.mean().item():.5f} std={e.std().item():.5f} l2_mean={l2:.5f} head8={head}")
                    else:
                        emitted_sec = round(total * (self._mel_hop / float(self._cos2_sr)), 3)
                        logging.info(f"[cos2.stream+text] uuid={uuid} finalize=True emitted_total={total} emitted_sec={emitted_sec} | spk=NA")
                except Exception:
                    # try to still include gt_info if available
                    gt_info = ""
                    try:
                        if gt_mel_len is not None:
                            if isinstance(gt_mel_len, torch.Tensor):
                                if gt_mel_len.numel() == 1:
                                    _v_list = [int(gt_mel_len.view(-1)[0].item())]
                                else:
                                    _v_list = [int(x) for x in gt_mel_len.view(-1).tolist()]
                            else:
                                _v_list = [int(gt_mel_len)]
                            gt_sec_list = [round(v * 256.0 / float(self.OUTPUT_SAMPLE_RATE), 3) for v in _v_list]
                            _v = _v_list[0] if len(_v_list) == 1 else _v_list
                            _s = gt_sec_list[0] if len(gt_sec_list) == 1 else gt_sec_list
                            gt_info = f" gt_mel_len={_v} gt_sec={_s}"
                    except Exception:
                        pass
                    emitted_sec = round(total * (self._mel_hop / float(self._cos2_sr)), 3)
                    logging.info(f"[cos2.stream+text] uuid={uuid} finalize=True emitted_total={total} emitted_sec={emitted_sec}{gt_info} | spk=ERR")
                self._mel_overlap_dict.pop(uuid, None)
                self._hift_cache_dict.pop(uuid, None)
                self._mel_total_len_dict.pop(uuid, None)

            # Vocoder with cache
            speech_24k, source = self._hift.inference(speech_feat=mel_for_vocoder, cache_source=cache_src)
            if not finalize:
                cache_src = source[:, :, -self._source_cache_len:]
                self._hift_cache_dict[uuid] = {
                    'source': cache_src,
                }
                # Do not drop tail samples; HiFT cache ensures continuity without duplication.
            else:
                cache_src = torch.zeros(batch_size, 1, 0, device=device)

            wav_chunks.append(speech_24k)

        wav_24k = torch.cat(wav_chunks, dim=-1)
        out_sr = self.OUTPUT_SAMPLE_RATE
        if self._cos2_sr != out_sr:
            wav = nemo_resample(wav_24k, self._cos2_sr, out_sr)
        else:
            wav = wav_24k

        return wav

    @torch.inference_mode()
    def offline_inference(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int64),
        prompt_feat: torch.Tensor = torch.zeros(1, 0, 80),
        embedding: torch.Tensor = torch.zeros(1, 192),
    ) -> torch.Tensor:
        """Non-streaming inference: run CosyVoice2 flow in finalize=True once and vocoder once.
        Returns wav at 22050 Hz for direct comparison with streaming path.
        """
        # debug: print speaker embedding stats for offline inference
        try:
            e = embedding.detach().float() if isinstance(embedding, torch.Tensor) else None
            if e is not None:
                l2 = torch.norm(e, dim=1).mean().item() if e.ndim == 2 and e.size(0) > 0 else float(torch.norm(e).item())
                head = e[0, :8].tolist() if e.ndim == 2 and e.size(0) > 0 else []
                logging.info(f"[cos2.offline] spk shape={list(e.shape)} mean={e.mean().item():.5f} std={e.std().item():.5f} l2_mean={l2:.5f} head8={head}")
            else:
                logging.info("[cos2.offline] spk=NA")
        except Exception:
            logging.info("[cos2.offline] spk=ERR")

        device = token.device
        # Ensure modules on correct device
        self._ensure_device(device)
        self._lazy_init_hift()
        if next(self._hift.parameters()).device != device:
            self._hift.to(device)

        batch_size = token.shape[0]
        wavs = []
        for batch_idx in range(batch_size):
            token_b = token[batch_idx : batch_idx + 1]
            ptok_b = prompt_token[batch_idx : batch_idx + 1] if prompt_token is not None and prompt_token.size(0) == batch_size else prompt_token
            pfeat_b = prompt_feat[batch_idx : batch_idx + 1] if prompt_feat is not None and prompt_feat.size(0) == batch_size else prompt_feat
            emb_b = embedding[batch_idx : batch_idx + 1] if embedding is not None and embedding.size(0) == batch_size else embedding

            # Option A: pre-upsample tokens so that encoder x2 gives desired token_mel_ratio (e.g., 4)
            upsample_factor = self._compute_upsample_factor()
            if upsample_factor > 1:
                token_b = token_b.repeat_interleave(upsample_factor, dim=1)
                ptok_b = ptok_b.repeat_interleave(upsample_factor, dim=1) if (ptok_b is not None and ptok_b.numel() > 0) else ptok_b

            # run flow once finalize=True to get full mel
            mel_b, _ = self.cos2_flow.inference(
                token=token_b,
                token_len=torch.tensor([token_b.shape[1]], dtype=torch.int32, device=device),
                prompt_token=ptok_b,
                prompt_token_len=torch.tensor([ptok_b.shape[1]], dtype=torch.int32, device=device),
                prompt_feat=pfeat_b,
                prompt_feat_len=torch.tensor([pfeat_b.shape[1]], dtype=torch.int32, device=device),
                embedding=emb_b,
                streaming=False,
                finalize=True,
            )
            # one-shot vocoder to avoid boundary artifacts
            speech_24k, _ = self._hift.inference(speech_feat=mel_b)
            # resample once at the end
            out_sr = self.OUTPUT_SAMPLE_RATE
            if self._cos2_sr != out_sr:
                wav = nemo_resample(speech_24k, self._cos2_sr, out_sr)
            else:
                wav = speech_24k
            wavs.append(wav)

        return torch.cat(wavs, dim=0)

