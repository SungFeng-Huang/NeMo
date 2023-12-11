# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import itertools
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.utils import logging, model_utils
from nemo.utils.decorators import experimental
from nemo.core.neural_types.elements import AudioSignal, FloatType, Index, IntType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.models.vits import VitsModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.tts.models.aligner import AlignerModel
from nemo.collections.tts.modules.voicebox_modules import (
    ConditionalFlowMatcherWrapper,
    VoiceBox,
    DurationPredictor,
    MelVoco,
    EncodecVoco,
    get_mask_from_lengths,
    interpolate_1d
)

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from nemo_text_processing.text_normalization.normalize import Normalizer


@experimental
class VoiceboxModel(TextToWaveform):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.tokenizer: BaseTokenizer | Tokenizer = None
        self.normalizer: Normalizer = None

        aligner = None
        dp_kwargs = {}
        # self.aligner: AlignerModel = None
        if cfg.get("nemo_aligner") and cfg.nemo_aligner.get("from_pretrained"):
            logging.info(cfg.nemo_aligner._target_)
            logging.info(get_class(cfg.nemo_aligner._target_))
            logging.info(cfg.nemo_aligner.from_pretrained)
            # aligner = AlignerModel.from_pretrained("tts_en_radtts_aligner")
            aligner = get_class(cfg.nemo_aligner._target_).from_pretrained(cfg.nemo_aligner.from_pretrained)
            aligner.freeze()

            self.tokenizer = aligner.tokenizer
            self.normalizer = aligner.normalizer
            self.text_normalizer_call_kwargs = aligner.text_normalizer_call_kwargs
            num_tokens = len(aligner.tokenizer.tokens)

            dp_kwargs.update({
                "tokenizer": self.tokenizer,
                "aligner": aligner
            })

        elif cfg.get("nemo_tokenizer"):
            # setup normalizer
            self.text_normalizer_call = None
            self.text_normalizer_call_kwargs = {}
            AlignerModel._setup_normalizer(self, cfg)

            # setup tokenizer
            AlignerModel._setup_tokenizer(self, cfg)    
            assert self.tokenizer is not None

            num_tokens = len(self.tokenizer.tokens)
            self.tokenizer_pad = self.tokenizer.pad
            dp_kwargs.update({
                "tokenizer": self.tokenizer,
            })

        elif cfg.get("mfa_tokenizer"):
            self.normalizer = None
            self.tokenizer = instantiate(cfg.mfa_tokenizer)
            num_tokens = self.tokenizer.vocab_size
            self.tokenizer_pad = self.tokenizer.pad_id
            dp_kwargs.update({
                "tokenizer": self.tokenizer,
            })

        elif cfg.get("tokenizer"):
            self.normalizer = None
            self.tokenizer = instantiate(cfg.tokenizer)
            num_tokens = self.tokenizer.vocab_size
            self.tokenizer_pad = self.tokenizer.pad_id
            dp_kwargs.update({
                "tokenizer": self.tokenizer,
            })

        super().__init__(cfg=cfg, trainer=trainer)

        # self.audio_enc_dec = instantiate(cfg.audio_enc_dec)
        # self.audio_enc_dec.freeze()

        self.duration_predictor = instantiate(
            cfg.duration_predictor,
            **dp_kwargs,
        )
        self.aligner = aligner

        self.voicebox: VoiceBox = instantiate(
            cfg.voicebox,
            num_cond_tokens=num_tokens
        )
        self.cfm_wrapper: ConditionalFlowMatcherWrapper = instantiate(
            cfg.cfm_wrapper,
            voicebox=self.voicebox,
            duration_predictor=self.duration_predictor,
            torchode_method_klass=get_class(cfg.cfm_wrapper.torchode_method_klass)
        )
        
    def prepare_data(self) -> None:
        """ Pytorch Lightning hook.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data

        The following code is basically for transcribed LibriLight.
        """
        from lhotse import CutSet
        from lhotse.serialization import load_manifest_lazy_or_eager
        from lhotse.recipes.utils import manifests_exist

        logging.info(f"mkdir -p raw_{self._cfg.manifests_dir}")
        os.makedirs("raw_" + self._cfg.manifests_dir, exist_ok=True)
        for subset in self._cfg.subsets:
            if not manifests_exist(subset, "raw_" + self._cfg.manifests_dir, ["cuts"], "libriheavy"):
                logging.info(f"Downloading {subset} subset.")
                os.system(f"wget -P raw_{self._cfg.manifests_dir} -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_{subset}.jsonl.gz")
            else:
                logging.info(f"Skipping download, {subset} subset exists.")

        # fix audio path prefix
        old_prefix="download/librilight"
        def change_prefix(cut):
            old_path = cut.recording.sources[0].source
            new_path = old_path.replace(old_prefix, self._cfg.corpus_dir)
            cut.recording.sources[0].source = new_path
            return cut
        
        def textgrid_exist(cut):
            cut_id = cut.id
            subset, spk = cut_id.split('/')[:2]
            f_id = f"{self.tokenizer.textgrid_dir}/{subset}/{spk}/{','.join(cut_id.split('/'))}.TextGrid"
            return os.path.exists(f_id)

        def parse_cut_mfa_textgrid(seg):
            from textgrid import TextGrid, IntervalTier
            from lhotse.supervision import AlignmentItem, SupervisionSet
            seg_id = seg.id
            subset, spk = seg_id.split('/')[:2]
            f_id = f"{self.tokenizer.textgrid_dir}/{subset}/{spk}/{','.join(seg_id.split('/'))}.TextGrid"
            tg = TextGrid()
            tg.read(f_id)
            phn_dur = []
            for tier in tg.tiers:
                if tier.name != "phones":
                    continue
                for interval in tier.intervals:
                    minTime = interval.minTime
                    maxTime = interval.maxTime
                    phoneme = interval.mark
                    if phoneme == "":
                        phoneme = "sil"
                    phn_dur.append(AlignmentItem(symbol=phoneme, start=minTime, duration=round(maxTime - minTime, 2)))
            assert len(phn_dur)
            new_sup_seg = seg.with_alignment("phone", phn_dur)
            return new_sup_seg

        logging.info(f"mkdir -p {self._cfg.manifests_dir}")
        os.makedirs(self._cfg.manifests_dir, exist_ok=True)
        for subset in self._cfg.subsets:
            manifest_path = os.path.join(self._cfg.manifests_dir, f"libriheavy_cuts_{subset}.jsonl.gz")
            if manifest_path not in [self._cfg.train_ds.manifest_filepath, self._cfg.validation_ds.manifest_filepath, self._cfg.test_ds.manifest_filepath]:
                continue
            if not os.path.exists(manifest_path):
                logging.info(f"Loading {subset} subset.")
                cuts = load_manifest_lazy_or_eager("raw_" + manifest_path, CutSet)
                logging.info(f"Filtering {subset} subset.")
                cuts = cuts.filter(lambda c: ',' not in c.id)
                cuts = cuts.filter(textgrid_exist)
                cuts = cuts.map_supervisions(parse_cut_mfa_textgrid)
                logging.info(f"Writing {subset} subset.")
                with CutSet.open_writer(
                    manifest_path, overwrite=False
                ) as cut_writer, tqdm(desc=f"Write {subset} subset") as progress:
                    for cut in cuts:
                        if cut_writer.contains(cut.id):
                            continue
                        cut_writer.write(cut)
                        progress.update()
                # cuts.to_file(manifest_path)
                del cuts
            else:
                logging.info(f"Skipping fix, {subset} subset exists.")

    def parse(self, str_input: str, **kwargs: Any) -> torch.tensor:
        if self.cfg.get("nemo_tokenizer"):
            assert all([k in ['normalize',] for k in kwargs.keys()])
            return VitsModel.parse(self, text=str_input **kwargs)
        
        tokens = self.tokenizer.text_to_ids(text=str_input)
        return torch.tensor(tokens).long().unsqueeze(0).to(self.device)

    def _setup_dataloader_from_config(self, config: Optional[Dict]) -> DataLoader[Any]:
        """Modified from https://github.com/pzelasko/NeMo/blob/feature/lhotse-integration/nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py#L129
        """
        from nemo.collections.asr.data.lhotse.dataloader import get_lhotse_dataloader_from_config
        from nemo.collections.tts.data.text_to_speech_lhotse import LhotseTextToSpeechDataset

        assert config.get("use_lhotse")

        # Note:
        #    Lhotse Dataset only maps CutSet -> batch of tensors, but does not actually
        #    contain any data or meta-data; it is passed to it by a Lhotse sampler for
        #    each sampler mini-batch.
        ds_kwargs = {}
        for kw in ["normalizer", "text_normalizer_call_kwargs", "tokenizer"]:
            if hasattr(self, kw):
                ds_kwargs[kw] = getattr(self, kw)

        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=LhotseTextToSpeechDataset(
                corpus_dir=self.cfg.corpus_dir,
                **ds_kwargs
            ),
        )
    
    def setup_training_data(self, train_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_training_data(self, train_data_config)
    
    def setup_validation_data(self, val_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_validation_data(self, val_data_config)

    def setup_test_data(self, test_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_test_data(self, test_data_config)

    # for inference
    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T_text'), TokenIndex()),
            "speakers": NeuralType(('B',), Index(), optional=True),
            "noise_scale": NeuralType(('B',), FloatType(), optional=True),
            "length_scale": NeuralType(('B',), FloatType(), optional=True),
            "noise_scale_w": NeuralType(('B',), FloatType(), optional=True),
            "max_len": NeuralType(('B',), IntType(), optional=True),
        }
    )
    def forward(
        self,
        cond = None,
        texts: Optional[List[str]] = None,
        phoneme_ids: Optional[Tensor] = None,
        cond_mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
    ):
        """
        Args

            - cond,

                    reference input audio

            - `texts`: Optional[List[str]] || `phoneme_ids`: Optional[Tensor]

                    input texts

            - cond_mask = None,

                    masking context audio

            - steps

                    ODE solver denoising steps

            - cond_scale
            
                    interpolate scaling for classifier-free inference guidance
        """
        audio = self.cfm_wrapper.sample(
            cond=cond,
            texts=texts,
            phoneme_ids=phoneme_ids,
            cond_mask=cond_mask,
            steps=steps,
            cond_scale=cond_scale,
            decode_to_audio=decode_to_audio
        )
        return audio

    
    def training_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT:
        # voicebox's sampling rate
        audio = batch["audio_24k"]
        audio_lens = batch["audio_lens_24k"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]

        # nemo aligner input
        audio_22050 = batch["audio_22050"]
        audio_lens_22050 = batch["audio_lens_22050"]
        tgt_len = audio.shape[1]

        # mfa tgt
        durations = batch.get("durations", None)


        audio_mask = get_mask_from_lengths(audio_lens)
        _, losses = self.cfm_wrapper.forward(
            x1=audio,
            mask=audio_mask,
            # semantic_token_ids=None,
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            dp_cond=None,
            durations=durations,
            cond=None,
            cond_mask=None,
            input_sampling_rate=None
        )
        # self.log("loss", loss, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])
        self.log_dict(losses, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])
        dp_loss = losses['d_pred_loss']
        align_loss = losses.get('align_loss', 0)
        bin_loss = losses.get('bin_loss', 0)
        vb_loss = losses['vb_loss']

        # if self.current_epoch < 10:
        #     align_loss = align_loss * 1e3
        #     bin_loss = bin_loss * 1e3
        # if self.current_epoch < 10:
        #     dp_loss = dp_loss * 0
        # if self.current_epoch < 10:
        #     vb_loss = vb_loss * 0

        loss = align_loss + bin_loss + dp_loss + vb_loss
        return loss
    
    def validation_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT | None:
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]

        audio_mask = get_mask_from_lengths(audio_lens)
        loss, losses = self.cfm_wrapper.forward(
            x1=audio,
            mask=audio_mask,
            # semantic_token_ids=None,
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            cond=None,
            cond_mask=None,
            input_sampling_rate=None
        )
        self.log_dict(losses, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])
        return loss

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        return list_of_models

    @typecheck(
        input_types={"tokens": NeuralType(('B', 'T_text'), TokenIndex(), optional=True),},
        output_types={"audio": NeuralType(('B', 'T_audio'), AudioSignal())},
    )
    def convert_text_to_waveform(self, *, tokens, speakers=None):
        audio = self(tokens=tokens, speakers=speakers)[0].squeeze(1)
        return audio