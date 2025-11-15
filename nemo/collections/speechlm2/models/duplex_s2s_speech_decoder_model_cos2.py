
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import torchaudio
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache, WhisperFeatureExtractor
import uuid
import os
from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.mos import MOS
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.modules.speech_tokenizer.utils import extract_speech_token
from nemo.collections.speechlm2.modules.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, setup_audio_codec, setup_speech_encoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging
import torch.distributed as dist
from nemo.collections.speechlm2.modules.flow_inference import AudioDecoder
import torch.nn.utils.rnn as rnn_utils
from librosa.filters import mel as librosa_mel_fn

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False):
    """Compute mel spectrogram with robust guards.
    Expects y shape [B, T] or [T]. Ensures non-empty and sufficient length for STFT+reflect.
    """
    # Ensure [B, T]
    if y.dim() == 1:
        y = y.unsqueeze(0)
    # Ensure floating type
    if not torch.is_floating_point(y):
        y = y.to(torch.float32)
    # Guard for empty/too-short inputs
    min_len = max(win_size, hop_size + 1)
    if y.numel() == 0 or y.shape[-1] < min_len:
        pad_needed = max(min_len - y.shape[-1], 0)
        if pad_needed > 0:
            y = torch.nn.functional.pad(y, (0, pad_needed))
        if y.numel() == 0:
            y = torch.zeros((1, min_len), dtype=torch.float32, device=y.device)

    # Optional sanity checks (avoid printing during training)
    if y.numel() > 0:
        _ = y.min(); _ = y.max()

    mel_basis = {}
    hann_window = {}  # pylint: disable=global-statement,global-variable-not-assigned
    device_key = str(y.device)
    mel_key = f"{str(fmax)}_{device_key}"
    if mel_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[mel_key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[device_key] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[device_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[mel_key], spec)

    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec


class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)

        # setup_audio_codec(self)
        self._codebook_size = 16384
        self._num_codebooks = 1


        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        # llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        # self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        # self.lm_head = llm.lm_head
        # self.embed_tokens = self.llm.embed_tokens
        # del self.llm.embed_tokens
        maybe_install_lora(self)

        # setup_speech_encoder(self)

        # self.speech_generation = TransformerARSpeechDecoder(
        #     speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
        #     lantent_dim=self.llm.config.hidden_size,
        #     num_audio_codebooks=self._num_codebooks,
        #     num_audio_tokens_per_codebook=self.speech_vocab_size,
        # )

        self.whispervq = WhisperVQEncoder.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            cache_dir='/hfcache',
        ).float().eval()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            cache_dir='/hfcache',
        )


        use_cos2 = getattr(self.cfg, "use_cos2_flow", False)
        if use_cos2:
            # Use CosyVoice2 causal streaming flow for training loss
            from nemo.collections.speechlm2.modules.cosyvoice2_adapter import CosyVoice2AudioDecoder
            cos2_cfg = getattr(self.cfg, "cos2_config_path", None)
            if not cos2_cfg or not os.path.isfile(cos2_cfg):
                raise FileNotFoundError(f"cos2_config_path not set or not found: {cos2_cfg}")
            # Provide fallback inference using legacy flow if pretrained_flow is set
            fallback_dir = getattr(self.cfg, "pretrained_flow", None)
            cos2_config_override = getattr(self.cfg, "cos2_config_override", None)
            # Build warm-start config with pretrained dirs from YAML
            warm_cfg = dict(self.cfg.get("warmstart", {}))
            cos2_dir = getattr(self.cfg, "cos2_dir", None)
            if cos2_dir:
                warm_cfg["cos2_dir"] = cos2_dir
            glm4_dir = getattr(self.cfg, "pretrained_flow", None)
            if glm4_dir:
                warm_cfg["glm4_dir"] = glm4_dir
            self.audio_decoder = CosyVoice2AudioDecoder(
                cos2_cfg,
                device=str(self.device),
                fallback_flow_dir=fallback_dir,
                cos2_config_override=cos2_config_override,
                warmstart_config=warm_cfg,
                stream_train_prob=float(self.cfg.get("cos2_stream_train_prob", 0.5)),
                use_text_context_train=bool(self.cfg.get("cos2_use_text_context_train", False)),
                token_overlap=int(self.cfg.get("cos2_token_overlap", 0)),
                is_debug=bool(self.cfg.get("is_debug", False)),
                print_per_n_chunk=int(self.cfg.get("print_per_n_chunk", 50)),
                stream_fixed_window_pad=bool(self.cfg.get("cos2_stream_fixed_window_pad", False)),
            )
        else:
            use_pf = getattr(self.cfg, "use_pretrained_flow", True)
            pf_dir = getattr(self.cfg, "pretrained_flow", None)
            # allow explicit vocoder path override
            hift_ckpt = getattr(self.cfg, "hift_ckpt", None)
            if hift_ckpt is None and pf_dir:
                hift_ckpt = os.path.join(pf_dir, "hift.pt")

            if use_pf:
                if not pf_dir:
                    raise ValueError("use_pretrained_flow=True but cfg.pretrained_flow is not set")
                flow_config = os.path.join(pf_dir, "config.yaml")
                flow_ckpt = os.path.join(pf_dir, "flow.pt")
                self.audio_decoder = AudioDecoder(flow_config, flow_ckpt, hift_ckpt)
            else:
                # random-init flow; vocoder optional
                self.audio_decoder = AudioDecoder(self.cfg.flow_config, None, hift_ckpt)

        # self.embed_audio_tokens = torch.nn.Embedding(16384, self.llm.config.hidden_size)
        # Speaker embedding extractor (192-D at 16 kHz)
        self.spk_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").eval()

        # Speaker embedding debug toggle: set via cfg.spk_emb_debug or env SPK_EMB_DEBUG=1
        self._spk_debug = bool(self.cfg.get("spk_emb_debug", False)) or bool(int(os.environ.get("SPK_EMB_DEBUG", "0")))



        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False


    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 4

    def _debug_log_spk(self, emb: torch.Tensor, where: str) -> None:
        if not getattr(self, "_spk_debug", False):
            return
        try:
            _rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            _rank = 0
        if _rank != 0:
            return
        if emb is None:
            logging.info(f"[SPK-EMB][{where}] None")
            return
        e = emb.detach().float()
        try:
            l2 = torch.norm(e, dim=1).mean().item() if e.ndim == 2 and e.size(0) > 0 else float(torch.norm(e).item())
        except Exception:
            l2 = float('nan')
        head = e[0, :8].tolist() if e.ndim == 2 and e.size(0) > 0 else []
        logging.info(
            f"[SPK-EMB][{where}] shape={list(e.shape)} mean={e.mean().item():.5f} std={e.std().item():.5f} min={e.min().item():.5f} max={e.max().item():.5f} l2_mean={l2:.5f} head8={head}"
        )

    # 统一在模型层覆盖 load_state_dict：
    # - 过滤旧检查点里与旧解码器/回退推理器相关的键，避免 Unexpected key(s)
    # - 使用 strict=False 防止缺失键造成报错（结构有变更）
    # - 不向上游返回 IncompatibleKeys，避免上层框架将其详细列表打印到控制台
    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        sd = dict(state_dict)
        if isinstance(sd, dict):
            # 注意：不要过滤 audio_decoder.flow.*，那是本项目训练得到的 CosyVoice2 flow 权重！
            drop_prefixes = (
                "audio_decoder.hift.",
                "audio_decoder._hift.",
                "audio_decoder._fallback_infer.",
            )
            for k in list(sd.keys()):
                if any(k.startswith(p) for p in drop_prefixes):
                    sd.pop(k, None)
        # 使用 strict=False，并吞掉返回值，防止上游打印详细 missing/unexpected 列表
        super().load_state_dict(sd, strict=False)
        return None

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size


    # 兼容旧检查点：如果检查点中包含了 audio_decoder.hift（GLM4-Voice/HiFiGAN 或 CosyVoice2 HiFT）权重，
    # 由于当前 CosyVoice2 的 HiFT 是在推理时懒加载（不参与训练/保存），需要在加载 state_dict 时忽略这些键。
    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", {})
        if not isinstance(state_dict, dict):
            return
        drop_prefixes = [
            "audio_decoder.hift.",
            "audio_decoder._hift.",
        ]
        remove_keys = [k for k in list(state_dict.keys()) if any(k.startswith(p) for p in drop_prefixes)]
        for k in remove_keys:
            state_dict.pop(k, None)
        # 同时兼容可能遗留的旧 flow cache/辅助模块键
        for k in list(state_dict.keys()):
            if k.startswith("audio_decoder._fallback_infer."):
                state_dict.pop(k, None)

        # 过滤旧检查点中的无关键，避免严格加载时报 Unexpected key(s)

        # 如果你希望从旧版（GLM4-Voice flow+HiFiGAN）检查点恢复，
        # 这些键在新版（CosyVoice2 适配器）中已不存在，需要在加载时剔除。



    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|


            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(self, input_embeds: Tensor, cache=None, input_audio_tokens=None, text_label=None, loss_mask=None) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        if loss_mask is not None:
            # This is training Mode
            loss_mask = loss_mask[:, :, -1].reshape(loss_mask.size(0), loss_mask.size(1))
            self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if text_label is not None:
            text_emb = self.embed_tokens(text_label)
        else:
            text_emb = self.embed_tokens(text_logits[:, -1].argmax(dim=-1).unsqueeze(dim=1))

        speech_input_hidden = text_emb * self.cfg.get("word_emb_weight", 1.0) + out['last_hidden_state']

        _, audio_logits = self.speech_generation(
            speech_input_hidden.transpose(0, 1), loss_mask, input_audio_tokens=input_audio_tokens
        )

        audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        准备模型输入数据，包括语音特征提取和token化
        这是flow matching训练的关键数据预处理步骤

        zhy:
        这里只用了target_audio，没有用source_audio
        batch的结构参考 NeMo/nemo/collections/speechlm2/data/s2s_dataset.py: __get_item__ 的返回值
        """
        # 1. 处理目标音频：去除padding（值为0的部分），并做最小长度兜底，避免空音频
        min_audio_len = int(0.25 * 22050)  # 250ms，可按需调整
        target_audio_list = []
        for sample in batch["target_audio"]:
            x = sample[sample != 0]
            if x.numel() < min_audio_len:
                pad = torch.zeros(min_audio_len - x.numel(), dtype=sample.dtype, device=sample.device)
                x = torch.cat([x, pad], dim=0)
            target_audio_list.append(x.unsqueeze(0))

        # 2. 提取梅尔频谱特征 - 用于flow matching的条件输入
        #    针对 CosyVoice2（HiFT 期望 24kHz/50fps，即 hop=480）与 GLM4-Voice（HiFiGAN 期望 22.05kHz/256）分别处理
        use_cos2 = getattr(self.cfg, "use_cos2_flow", False)
        target_mel_list = []
        if use_cos2:
            # 将 22.05kHz 波形重采样到 24kHz，再用 CosyVoice2 的梅尔参数提取（与 HiFT 训练对齐）
            mel_audio_list_24k = [resample(x, 22050, 24000) for x in target_audio_list]
            for wav_24k in mel_audio_list_24k:
                mel_24k = mel_spectrogram(
                    wav_24k,
                    n_fft=1920,
                    num_mels=80,
                    sampling_rate=24000,
                    hop_size=480,
                    win_size=1920,
                    fmin=0,
                    fmax=8000,
                    center=False,
                )
                target_mel_list.append(mel_24k)
        else:
            # 与 GLM4-Voice HiFi-GAN 训练域一致：22.05kHz, hop=256
            for wav_22k in target_audio_list:
                mel_22k = mel_spectrogram(
                    wav_22k,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=22050,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=8000,
                    center=False,
                )
                target_mel_list.append(mel_22k)

        # 3. 计算每个样本的频谱长度
        speech_feat_lens = torch.tensor([x.shape[2] for x in target_mel_list], device=target_mel_list[0].device)
        speech_audio_lens = torch.tensor([x.shape[1] for x in target_audio_list], device=target_audio_list[0].device)   # 22050Hz

        # 4. 将不同长度的频谱特征padding到相同长度
        speech_feat = torch.nn.utils.rnn.pad_sequence(
            [x.squeeze(0).permute(1, 0) for x in target_mel_list],
            batch_first=True
        ).permute(0, 2, 1)  # 最终形状: [batch, mel_bins, time_frames]

        # 5. 使用fp32精度进行音频重采样（避免bfloat16精度问题）
        with fp32_precision():
            # 从22050Hz重采样到16000Hz，匹配WhisperVQ的输入要求
            target_audio_16k_list = [resample(x, 22050, 16000) for x in target_audio_list]

            # 6. 使用WhisperVQ提取离散语音token
            speech_tokens = extract_speech_token(
                self.whispervq,  # WhisperVQ编码器
                self.feature_extractor,  # 特征提取器
                [(target_audio_16k_list[i], 16000) for i in range(len(target_audio_16k_list))],
            )

        # 7. 处理speech tokens的长度和padding
        speech_token_len = torch.tensor([len(seq) for seq in speech_tokens], dtype=torch.long, device=self.device)
        max_len = speech_token_len.max().item()
        padded_tensor = torch.zeros(len(speech_tokens), max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(speech_tokens):
            padded_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

        speech_tokens = padded_tensor

        # if dist.get_rank() == 0:
        #     logging.info(f"speech_feat shape: {speech_feat.shape}, value: {speech_feat}\n")
        #     logging.info(f"speech_feat_lens shape: {speech_feat_lens.shape}, value: {speech_feat_lens}\n")
        #     logging.info(f"speech_tokens shape: {speech_tokens.shape}, value: {speech_tokens}\n")
        #     logging.info(f"speech_token_len shape: {speech_token_len.shape}, value: {speech_token_len}\n")

        #     logging.info(f"embedding shape: {torch.zeros(speech_tokens.size(0), 192).shape}\n")

        # 8. 准备 text tokens 供训练/验证时“文本窗口”使用
        texts = batch.get("target_texts", []) if isinstance(batch, dict) else []
        if isinstance(texts, (list, tuple)) and len(texts) > 0:
            token_seqs = [self.tokenizer.text_to_ids(t) for t in texts]
            max_len_txt = max(1, max((len(seq) for seq in token_seqs), default=1))
            pad_id = self.text_pad_id
            text_tokens = torch.full((len(token_seqs), max_len_txt), pad_id, dtype=torch.long, device=self.device)
            text_token_len = torch.zeros(len(token_seqs), dtype=torch.long, device=self.device)
            for i, seq in enumerate(token_seqs):
                if len(seq) > 0:
                    text_tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                    text_token_len[i] = len(seq)
        else:
            text_tokens = torch.zeros(speech_tokens.size(0), 0, dtype=torch.long, device=self.device)
            text_token_len = torch.zeros(speech_tokens.size(0), dtype=torch.long, device=self.device)
        # 9. Extract speaker embeddings (192-D) from TARGET audio at 16 kHz using Titanet
        with torch.no_grad():
            try:
                self.spk_model.to(self.device)
            except Exception:
                pass
            audio_16k = rnn_utils.pad_sequence([x.squeeze(0) for x in target_audio_16k_list], batch_first=True).to(self.device)
            audio_16k_lens = torch.tensor([x.shape[1] for x in target_audio_16k_list], dtype=torch.long, device=self.device)
            _, spk_emb = self.spk_model.forward(input_signal=audio_16k, input_signal_length=audio_16k_lens)

        self._debug_log_spk(spk_emb, f"prepare_inputs:{'train' if self.training else 'eval'}")

        # logging.info(f"[prepared_inputs] texts: {texts}")
        # logging.info(f"[prepared_inputs] text_tokens: {text_tokens}")

        return {
            "speech_audio": target_audio_list,
            "speech_audio_lens": speech_audio_lens,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_lens,
            "speech_token": speech_tokens,
            "speech_token_len": speech_token_len,
            "embedding": spk_emb,
            # 新增：文本 token 提供给 flow 训练时的文本窗口
            "text_tokens": text_tokens,
            "text_token_len": text_token_len,
        }


        # zhy: 在flow matching训练里没用，暂时注释掉
        '''

        """Prepares input tensors for the model."""
        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )


        #
        # speech_feat = mel_spectrogram(batch["target_audio"]) # 1
        # speech_feat_lens = (batch["target_audio_lens"] - 256) // 256 + 1  # 2
        # 【batch, 80, n_feat】

        target_tokens = batch["target_tokens"]

        with fp32_precision():  # resample is fragile to bfloat16 default dtype
            resampled_audio = resample(batch["target_audio"], 22050, 16000)

            target_speech_tokens = extract_speech_token(
                self.whispervq,
                self.feature_extractor,
                [(resampled_audio[i].unsqueeze(0), 16000) for i in range(batch["target_audio"].shape[0])],
            )
        # import pdb; pdb.set_trace()
        target_speech_tokens = torch.tensor(target_speech_tokens, dtype=torch.long, device=self.device)
        # speech_tokens = speech_tokens[:, :max(speech_tokens_lens)] # 4



        min_len = min(source_encoded.shape[1], target_speech_tokens.shape[1], target_tokens.shape[1])
        source_encoded = source_encoded[:, :min_len]
        target_speech_tokens = target_speech_tokens[:, :min_len].unsqueeze(dim=-1)
        target_tokens = target_tokens[:, :min_len]
        source_encoded_lens = torch.clamp_(source_encoded_lens, max=min_len)



        btt = target_tokens[..., None]

        target_speech_tokens = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_speech_tokens)
        target_speech_tokens = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_speech_tokens)

        target_speech_tokens = torch.cat(
            [
                torch.full(
                    [target_speech_tokens.shape[0], 1, target_speech_tokens.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_speech_tokens[:, :-1],
            ],
            dim=1,
        )

        input_ids = torch.cat([target_speech_tokens, target_tokens[..., None]], dim=-1)

        text_inputs = input_ids[:, :-1, -1]
        text_labels = input_ids[:, 1:, -1]
        audio_inputs = input_ids[:, :-1, :1]
        audio_labels = input_ids[:, 1:, :1]

        input_embeds = self.embed_tokens(text_inputs)


        user_stream = source_encoded[:, :-1] * self.cfg.get("duplex_user_emb_weight", 1.0)
        input_embeds.add_(user_stream)

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )


        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": source_encoded_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
            "speech_feat": speech_feat,
            # for flow matching training
            "speech_feat_len": speech_feat_lens,
            "speech_token": speech_tokens,
            "speech_token_len": speech_tokens_lens,
            "embedding": torch.zeros(speech_tokens.size(0), 192).to(self.device)
        }

        '''


    def cal_acc(self, pad_outputs, pad_targets, ignore_label):
        pad_pred = pad_outputs.argmax(-1)
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
        denominator = torch.sum(mask)
        return (numerator / denominator).detach().item()

    def training_step(self, batch: dict, batch_idx: int):
        """
        Flow matching训练的核心步骤
        """
        # 准备输入数据（包括梅尔频谱、语音token等）
        inputs = self.prepare_inputs(batch)

        # Flow matching训练的核心：
        # - inputs包含条件信息（梅尔频谱）和目标（语音token）
        # - audio_decoder.flow执行flow matching的前向传播和损失计算
        loss = self.audio_decoder.flow(inputs, self.device)

        # 记录训练指标
        ans = {
            "loss": loss['loss'],  # Flow matching损失
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
        }
        self.log_dict(ans, on_step=True)
        return ans
        # import pdb; pdb.set_trace()


        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            text_label=inputs['text_labels'],
            loss_mask=inputs["loss_mask"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["text_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            audio_loss = torch.nn.functional.cross_entropy(
                forward_outputs["audio_logits"].flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                inputs["audio_labels"].flatten(0, 2),
                reduction="sum",
            ) / (num_frames * self._num_codebooks)
        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        text_acc = self.cal_acc(forward_outputs["text_logits"].flatten(0, 1), inputs["text_labels"].flatten(0, 1), 0 )
        audio_acc = self.cal_acc(forward_outputs["audio_logits"].flatten(0, 1), inputs["audio_labels"].flatten(0, 1), -1)

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "text_acc": text_acc,
            "audio_acc": audio_acc,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        # self.bleu = BLEU().reset()
        self.mos = MOS().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        # bleu = self.bleu.compute()
        # for k, m in bleu.items():
        #     self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        mos = self.mos.compute()
        for k, m in mos.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)


    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        outputs = {
            "loss": 0.0,
            "audio_list": [],
            "image_list": [],
        }
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            inputs = self.prepare_inputs(dataset_batch)

            batch = inputs['speech_token'].shape[0]


            this_uuid = str(uuid.uuid4())

            prompt_speech_feat = torch.zeros(batch, 0, 80).to(self.device)
            flow_prompt_speech_token = torch.zeros(batch, 0, dtype=torch.int64).to(self.device)
            spk_emb = inputs.get('embedding', torch.zeros(batch, 192, device=self.device))
            # debug: print speaker embedding during validation/inference
            self._debug_log_spk(spk_emb, "validation:spk_emb")


            with fp32_precision(), torch.no_grad():

                # Run both streaming and non-streaming inference for AB comparison
                use_stream = bool(getattr(self.cfg, 'streaming_infer', False)) and hasattr(self.audio_decoder, 'stream_inference')
                response_speech_stream = None
                if use_stream:
                    # choose between vanilla streaming or text-context streaming via config flag
                    use_text_ctx = bool(getattr(self.cfg, 'cos2_use_text_context_infer', False) and getattr(self.audio_decoder, 'use_cross_text_attn', False))
                    if use_text_ctx and hasattr(self.audio_decoder, 'stream_inference_with_text'):
                        texts = dataset_batch.get("target_texts", [])
                        # logging.info(f"\n[validation_step] stream_inference_with_text true")
                        # logging.info(f"[validation_step] target texts: {texts}")

                        # Build text tokens from target_texts on-the-fly using NeMo tokenizer
                        text_tokens = None
                        if isinstance(texts, (list, tuple)) and len(texts) > 0:
                            token_seqs = [self.tokenizer.text_to_ids(t) for t in texts]
                            # logging.info(f"[validation_step] text tokens: {token_seqs}")
                            max_len_txt = max(1, max((len(seq) for seq in token_seqs), default=1))
                            pad_id = self.text_pad_id
                            text_tokens = torch.full((batch, max_len_txt), pad_id, dtype=torch.long, device=self.device)
                            for i, seq in enumerate(token_seqs):
                                if len(seq) > 0:
                                    text_tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

                        if text_tokens is not None and isinstance(text_tokens, torch.Tensor) and text_tokens.numel() > 0:
                            # logging.info(f"[validation_step] stream_inference_with_text")
                            # logging.info(f"[validation_step] text_tokens: {text_tokens}")
                            # logging.info(f"[validation_step] spk_emb: {spk_emb}")
                            response_speech_stream = self.audio_decoder.stream_inference_with_text(
                                inputs['speech_token'],
                                uuid=this_uuid,
                                text_tokens=text_tokens,
                                prompt_token=flow_prompt_speech_token.to(self.device),
                                prompt_feat=prompt_speech_feat.to(self.device),
                                embedding=spk_emb,
                                gt_mel_len=inputs['speech_feat_len'],
                            )
                        else:
                            # logging.info(f"[validation_step] stream_inference")
                            response_speech_stream = self.audio_decoder.stream_inference(
                                inputs['speech_token'],
                                uuid=this_uuid,
                                prompt_token=flow_prompt_speech_token.to(self.device),
                                prompt_feat=prompt_speech_feat.to(self.device),
                                embedding=spk_emb,
                                gt_mel_len=inputs['speech_feat_len'],
                            )
                    else:
                        # logging.info(f"\n[validation_step] stream_inference_with_text false")
                        response_speech_stream = self.audio_decoder.stream_inference(
                            inputs['speech_token'],
                            uuid=this_uuid,
                            prompt_token=flow_prompt_speech_token.to(self.device),
                            prompt_feat=prompt_speech_feat.to(self.device),
                            embedding=spk_emb,
                            gt_mel_len=inputs['speech_feat_len'],
                        )
                # Always run offline (non-streaming) inference for comparison
                if hasattr(self.audio_decoder, 'offline_inference'):
                    response_speech_offline = self.audio_decoder.offline_inference(
                        inputs['speech_token'],
                        prompt_token=flow_prompt_speech_token.to(self.device),
                        prompt_feat=prompt_speech_feat.to(self.device),
                        embedding=spk_emb,
                    )
                else:
                    # fallback to non-streaming token2wav
                    response_speech_offline, _ = self.audio_decoder.token2wav(inputs['speech_token'],
                                                                               uuid=this_uuid,
                                                                               prompt_token=flow_prompt_speech_token.to(self.device),
                                                                               prompt_feat=prompt_speech_feat.to(self.device),
                                                                               embedding=spk_emb,
                                                                               finalize=True)


                # Save both versions for AB debugging
                target_audio_list = inputs['speech_audio']
                speech_audio_lens = inputs['speech_audio_lens']
                os.makedirs(self.cfg.get('audio_save_path'), exist_ok=True)
                for i in range(batch):
                    _base = f"{name}_{dataloader_idx}_{batch_idx}_{i}_{dataset_batch['sample_id'][i]}"
                    base = f"{self.cfg.audio_save_path}/{_base}"

                    target_audio_i = target_audio_list[i]
                    torchaudio.save(base + "_0_target.wav", target_audio_i.cpu(), 22050)
                    outputs['audio_list'].append(dict(id=f"{dataloader_idx}_{batch_idx}_{i}_0_target", data=target_audio_i.cpu(), sample_rate=22050, filepath=f"{_base}_0_target.wav"))
                    if response_speech_stream is not None:
                        response_speech_stream_i = response_speech_stream[i, :speech_audio_lens[i]]
                        torchaudio.save(base + "_1_stream.wav", response_speech_stream_i.unsqueeze(0).cpu(), 22050)
                        outputs['audio_list'].append(dict(id=f"{dataloader_idx}_{batch_idx}_{i}_1_stream", data=response_speech_stream_i.unsqueeze(0).cpu(), sample_rate=22050, filepath=f"{_base}_1_stream.wav"))
                    if response_speech_offline is not None:
                        response_speech_offline_i = response_speech_offline[i, :speech_audio_lens[i]]
                        torchaudio.save(base + "_2_offline.wav", response_speech_offline_i.unsqueeze(0).cpu(), 22050)
                        outputs['audio_list'].append(dict(id=f"{dataloader_idx}_{batch_idx}_{i}_2_offline", data=response_speech_offline_i.unsqueeze(0).cpu(), sample_rate=22050, filepath=f"{_base}_2_offline.wav"))

                # Ground-truth mel reconstruction via HiFT (quick vocoder sanity check)
                try:
                    import torch.nn.functional as F
                    # Prepare GT mel for HiFT: expect [B, 80, T]
                    mel_gt = inputs['speech_feat']  # could be [B, T, 80] or [B, 80, T]
                    if mel_gt.ndim != 3:
                        raise RuntimeError(f"speech_feat ndim={mel_gt.ndim} not supported")
                    if mel_gt.shape[1] == 80:
                        # already [B, 80, T]
                        pass
                    elif mel_gt.shape[2] == 80:
                        # [B, T, 80] -> [B, 80, T]
                        mel_gt = mel_gt.transpose(1, 2).contiguous()
                    else:
                        raise RuntimeError(f"speech_feat shape={tuple(mel_gt.shape)} not compatible with HiFT (need 80 mel channels)")
                    mel_gt = mel_gt.to(self.device).float()

                    # Ensure vocoder (HiFi-GAN/HiFT) exists and on correct device
                    vocoder = getattr(self.audio_decoder, '_hift', None)
                    if vocoder is None:
                        vocoder = getattr(self.audio_decoder, 'hift', None)
                    if vocoder is None and hasattr(self.audio_decoder, '_lazy_init_hift'):
                        # CosyVoice2 adapter lazy init
                        self.audio_decoder._lazy_init_hift()
                        vocoder = getattr(self.audio_decoder, '_hift', None)
                    if vocoder is None:
                        raise AttributeError("No vocoder found on audio_decoder (expected '_hift' or 'hift')")

                    hift_dev = next(vocoder.parameters()).device
                    if hift_dev != self.device:
                        vocoder.to(self.device)

                    # Optional debug
                    if os.environ.get('COS2_VAL_DEBUG', '0') == '1':
                        logging.info(f"[val][reconGT] mel_gt shape={tuple(mel_gt.shape)} mean/std={mel_gt.mean().item():.4f}/{mel_gt.std().item():.4f}")

                    # Run vocoder with GT mel (assumed 50 fps -> hop=480). If dataset mel fps>50, it will sound slowed.
                    def _infer_vocoder(voc, mel):
                        return voc.inference(speech_feat=mel)

                    wav24k_gt, _ = _infer_vocoder(vocoder, mel_gt)

                    # Try temporal downsampling (avg-pool) by 2x and 4x to test fps mismatch quickly
                    mel_gt_x2 = F.avg_pool1d(mel_gt, kernel_size=2, stride=2)
                    mel_gt_x4 = F.avg_pool1d(mel_gt, kernel_size=4, stride=4)
                    wav24k_gt_x2, _ = _infer_vocoder(vocoder, mel_gt_x2)
                    wav24k_gt_x4, _ = _infer_vocoder(vocoder, mel_gt_x4)

                    # Resample to 22050 for saving alongside predictions
                    orig_sr = (
                        getattr(self.audio_decoder, '_cos2_sr', None)
                        or getattr(vocoder, 'sample_rate', None)
                        or 24000
                    )
                    wav22050_gt = resample(wav24k_gt, orig_sr, 22050)
                    wav22050_gt_x2 = resample(wav24k_gt_x2, orig_sr, 22050)
                    wav22050_gt_x4 = resample(wav24k_gt_x4, orig_sr, 22050)

                    # Save a few examples per batch (all by default)
                    for i in range(batch):
                        _base = f"{name}_{dataloader_idx}_{batch_idx}_{i}_{dataset_batch['sample_id'][i]}"
                        base = f"{self.cfg.audio_save_path}/{_base}"

                        wav22050_gt_i = wav22050_gt[i, :speech_audio_lens[i]]
                        torchaudio.save(base + "_3_reconGT.wav", wav22050_gt_i.unsqueeze(0).cpu(), 22050)
                        outputs['audio_list'].append(dict(id=f"{dataloader_idx}_{batch_idx}_{i}_3_reconGT", data=wav22050_gt_i.unsqueeze(0).cpu(), sample_rate=22050, filepath=f"{_base}_3_reconGT.wav"))

                        wav22050_gt_x2_i = wav22050_gt_x2[i, :speech_audio_lens[i]//2]
                        torchaudio.save(base + "_4_reconGT_x2.wav", wav22050_gt_x2_i.unsqueeze(0).cpu(), 22050)
                        # outputs['audio_list'].append(dict(id=f"{dataloader_idx}_{batch_idx}_{i}_4_reconGT_x2", data=wav22050_gt_x2[i].unsqueeze(0).cpu(), sample_rate=22050, filepath=f"{_base}_4_reconGT_x2.wav"))

                        wav22050_gt_x4_i = wav22050_gt_x4[i, :speech_audio_lens[i]//4]
                        torchaudio.save(base + "_5_reconGT_x4.wav", wav22050_gt_x4_i.unsqueeze(0).cpu(), 22050)
                        # outputs['audio_list'].append(dict(id=f"{dataloader_idx}_{batch_idx}_{i}_5_reconGT_x4", data=wav22050_gt_x4[i].unsqueeze(0).cpu(), sample_rate=22050, filepath=f"{_base}_5_reconGT_x4.wav"))

                        if os.environ.get('COS2_VAL_DEBUG', '0') == '1':
                            logging.info(f"[val][reconGT] saved: {base}_reconGT(.wav, _x2.wav, _x4.wav)")
                        # logging.info(f"[val][reconGT] saved: {self.cfg.audio_save_path}/{name}_{i}_{dataset_batch['sample_id'][i]}_reconGT.wav")
                except Exception as e:
                    logging.info(f"[val][reconGT] failed: {e}")


                # Choose which to feed to metrics (keep previous behavior: prefer streaming if enabled)
                response_speech = response_speech_stream if response_speech_stream is not None else response_speech_offline
                pred_audios = resample(response_speech, 22050, 16000)

                # use speech_audio_lens to force the audio length to be the same as the target audio length
                self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=pred_audios,
                    pred_audio_lens=torch.tensor(speech_audio_lens / 22050 * 16000).repeat(batch).to(torch.long),
                )

                self.mos.update(
                    name=name,
                    pred_audios=pred_audios,
                    tmp_dir=os.path.join(self.cfg.get('audio_save_path'), "tmp"),
                )


            # results = self.offline_inference(
            #     dataset_batch["source_audio"],
            #     dataset_batch["source_audio_lens"],
            #     decode_audio=True,
            # )
            #
            # with fp32_precision():  # resample is fragile to bfloat16 default dtype
            #     predicted_audio = resample(results['audio'], 22050, 16000)
            #     self.asr_bleu.update(
            #         name=name,
            #         refs=dataset_batch["target_texts"],
            #         pred_audio=predicted_audio,
            #         pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
            #     )
            # self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            #
            # if self.cfg.get('audio_save_path') is not None and dist.get_rank() == 0:
            #
            #
            #     os.makedirs(self.cfg.get('audio_save_path'), exist_ok=True)
            #     logging.info(f"The shape of generated speech: {predicted_audio.shape}")
            #     for i in range(len(predicted_audio)):
            #         pred_audio = predicted_audio[i]
            #         user_audio = dataset_batch["source_audio"][i]
            #
            #         T1, T2 = pred_audio.shape[0], user_audio.shape[0]
            #         max_len = max(T1, T2)
            #         pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
            #         user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)
            #
            #         result_audio = pred_audio_padded + user_audio_padded
            #
            #         torchaudio.save(f"{self.cfg.audio_save_path}/{name}_{i}_{dataset_batch['sample_id'][i]}.wav",
            #                         result_audio.unsqueeze(0).float().cpu(),
            #                         16000)
            #
            # dist.barrier()
        return outputs

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        input_embeds, lengths = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
        )

        # source_audio_tokens = extract_speech_token(
        #     self.whispervq,
        #     self.feature_extractor,
        #     [(input_signal[i].unsqueeze(0), 16000) for i in range(input_signal.shape[0])],
        # )

        # source_audio_tokens = torch.tensor(source_audio_tokens, dtype=torch.long, device=self.device)
        # source_audio_embeds = self.embed_audio_tokens(source_audio_tokens)
        # T_min = min(input_embeds.shape[1], source_audio_embeds.shape[1])

        # input_embeds = input_embeds[:, :T_min]
        # source_audio_embeds = source_audio_embeds[:, :T_min]


        B, T_local, H = input_embeds.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=input_embeds.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame = input_embeds[:, T_local - 1 : T_local, :]  # (B,1,H)
                pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                input_embeds = torch.cat([input_embeds, pad], dim=1)
        else:
            T = T_local

        # Apply channel weight

        input_embeds = input_embeds * self.cfg.get("duplex_user_emb_weight", 1.0)

        # This cache is for self.llm
        cache = DynamicCache()
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()



        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(input_embeds[:, :1], cache=cache, input_audio_tokens=first_audio, loss_mask=None)
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(input_embeds[:, t : t + 1], cache=ans["cache"], input_audio_tokens=current_audio)
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }


        if decode_audio:
            # self.load_flow_decoder()
            with fp32_precision(), torch.no_grad():
                this_uuid = str(uuid.uuid4())


                prompt_speech_feat = torch.zeros(input_embeds.shape[0], 0, 80).to(self.device)
                flow_prompt_speech_token = torch.zeros(input_embeds.shape[0], 0, dtype=torch.int64).to(self.device)
                # Use real speaker embeddings from the provided input_signal; resample to 16 kHz for Titanet
                try:
                    self.spk_model.to(self.device)
                except Exception:
                    pass
                src_sr = int(getattr(self.cfg, 'source_sample_rate', 16000))
                with fp32_precision():
                    sig_16k = input_signal.to(self.device)
                    if src_sr != 16000:
                        sig_16k = resample(sig_16k, src_sr, 16000)
                    lens_16k = input_signal_lens.to(self.device)
                    if src_sr != 16000:
                        lens_16k = torch.clamp((lens_16k.to(torch.float32) * (16000.0 / src_sr)).round().to(torch.long), min=1)

                _, spk_emb = self.spk_model.forward(input_signal=sig_16k, input_signal_length=lens_16k)

                self._debug_log_spk(spk_emb, "offline_infer:decode_audio")

                flow_input_token = gen_audio[:,:,0]
                flow_input_token[flow_input_token >16383] = 0
                #
                response_speech, _ = self.audio_decoder.token2wav(flow_input_token,
                                                                  uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(self.device),
                                                                  prompt_feat=prompt_speech_feat.to(self.device),
                                                                  embedding=spk_emb,
                                                                  finalize=True)

                # response_speech = self.audio_decoder.stream_inference(flow_input_token,
                #                                                   this_uuid,
                #                                                   flow_prompt_speech_token.to(self.device),
                #                                                   prompt_speech_feat.to(self.device),
                #                                                   spk_emb,
                #                                                   )
                ans["audio"] = response_speech
                ans["audio_len"] = torch.tensor(response_speech.shape[1]).unsqueeze(0).repeat(input_embeds.shape[0]).to(self.device)


        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)

    def configure_callbacks(self):
        # self.log_config = self.cfg.get('log_config', None)
        # if not self.log_config:
        #     return []

        from nemo.collections.speechlm2.parts.utils.callbacks import LoggingCallback

        log_dir = None # no multiple save for now
        log_callback = LoggingCallback(
            generators=None,
            data_loader=None,
            log_epochs=None,
            epoch_frequency=None,
            output_dir=log_dir,
            loggers=self.trainer.loggers,
            log_tensorboard=True,
            log_wandb=True,
        )

        return [log_callback]