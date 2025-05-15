from typing import Dict, Optional, Tuple, Union

import einops
import hydra
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.audio.models.audio_to_audio import AudioToAudioModel
from nemo.collections.audio.models.enhancement import FlowMatchingAudioToAudioModel
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.data.vocoder_web_dataset import create_vocoder_dataset
# from nemo.collections.tts.models import AudioCodecModel
from nemo.core.neural_types import AudioSignal, LengthsType, LossType, NeuralType
from nemo.core.classes import NeuralModule, typecheck
from nemo.utils import logging



class AudioToCodec(NeuralModule):
    """Transform a batch of input multi-channel signals into a batch of
    codec-based spectrograms.

    Args:
        codec: codec model to be used for the transformation
    """

    def __init__(
        self,
        restore_codec_from_path: str = None,
        feature_type: str = "quantized",
    ):
        super().__init__()
        if restore_codec_from_path is not None:
            self.codec = AudioCodecModel.restore_from(
                restore_path=restore_codec_from_path, map_location="cpu", strict=True
            ).eval()
        else:
            raise ValueError("Either restore_from_path or cfg must be provided to initialize the codec model.")
        self.codec.freeze()

        self.feature_type = feature_type
        if self.feature_type not in ["quantized", "postquantized", "prequantized"]:
            raise ValueError(f"Unknown feature type {self.feature_type}. Supported types are quantized, postquantized, prequantized")
        self.num_groups = self.codec.num_groups
        if self.feature_type == "quantized":
            self.feature_dim = self.codec.num_groups
        elif self.feature_type in ["postquantized", "prequantized"]:
            self.feature_dim = self.codec.codebook_dim_per_group * self.codec.num_groups

        logging.debug('Initialized %s with:', self.__class__.__name__)
        logging.debug('\tcodec:           %s', self.codec)
        logging.debug('\tsample_rate:     %s', self.codec.sample_rate)
        logging.debug('\tfeature_type:    %s', self.feature_type)
        logging.debug('\tfeature_dim:     %s', self.feature_dim)
        logging.debug('\tnum_groups:      %s', self.num_groups)

    @property
    def win_length(self) -> int:
        # TODO: check
        return self.codec.samples_per_frame

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, input: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of C-channel input signals
        into a batch of codec-based spectrograms.

        Args:
            input: Time-domain input signal with C channels, shape (B, C, T)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Output codec features with F subbands and N time frames, shape (B, C, F, N)
            and output length with shape (B,).
        """
        B, T = input.size(0), input.size(-1)
        input = input.view(B, -1, T)

        # codec output (B, C, F, N)
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
            if self.feature_type == "prequantized":
                # encoded (B, codebook_dim_per_group * num_groups, T)
                # encoded_len (B)
                encoded, encoded_len = self.encode_audio(audio=input, audio_len=input_length)
                return encoded, encoded_len
            
            elif self.feature_type in ["quantized", "postquantized"]:
                # FSD uses rounded scalar to determin code_id
                # gen_tokens (B, num_groups, T)
                gen_tokens, gen_tokens_lens = self.codec.encode(audio=input, audio_len=input_length)

                if self.feature_type == "quantized":
                    return gen_tokens, gen_tokens_lens

                elif self.feature_type == "postquantized":
                    # get dequantized representation
                    # dequantized (B, codebook_dim_per_group * num_groups, T)
                    dequantized = self.codec.dequantize(tokens=gen_tokens, tokens_len=gen_tokens_lens)
                    return dequantized, gen_tokens_lens
                
            else:
                raise ValueError(f"Unknown feature type {self.feature_type}. Supported types are quantized, postquantized, prequantized")


class CodecFlowMatchingAudioToAudioModel(FlowMatchingAudioToAudioModel):
    """CodecFlowMatchingAudioToAudio model that uses codec as the encoder to extract features as the condition of the estimator.

    The model consists of the following blocks:
        - encoder: transforms input multi-channel audio signal into an encoded representation (analysis transform)
        - estimator: neural model, estimates a score for the diffusion process
        - flow: ordinary differential equation (ODE) defining a flow and a vector field.
        - sampler: sampler for the inference process, estimates coefficients of the target signal
        - decoder: transforms sampler output into the time domain (synthesis transform)
        - ssl_pretrain_masking: if it is defined, perform the ssl pretrain masking for self reconstruction in the training process
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # AudioToAudioModel.__init__(self, cfg=cfg, trainer=trainer)
        # self.sample_rate = self._cfg.sample_rate

        # # Setup processing modules
        # self.encoder = self.from_config_dict(self._cfg.encoder)
        # self.decoder = self.from_config_dict(self._cfg.decoder)

        # Codec
        self.codec = self.from_config_dict(self._cfg.codec)
        super().__init__(cfg=cfg, trainer=trainer)

        # # Neural estimator
        # self.estimator = self.from_config_dict(self._cfg.estimator)

        # # Flow
        # self.flow = self.from_config_dict(self._cfg.flow)

        # # Sampler
        # self.sampler = hydra.utils.instantiate(self._cfg.sampler, estimator=self.estimator)

        # # probability that the conditional input will be feed into the
        # # estimator in the training stage
        # self.p_cond = self._cfg.get('p_cond', 1.0)

        # # Self-Supervised Pretraining
        # if self._cfg.get('ssl_pretrain_masking') is not None:
        #     logging.debug('SSL-pretrain_masking is found and will be initialized')
        #     self.ssl_pretrain_masking = self.from_config_dict(self._cfg.ssl_pretrain_masking)
        # else:
        #     self.ssl_pretrain_masking = None

        # # Normalization
        # self.normalize_input = self._cfg.get('normalize_input', False)

        # # Metric evaluation
        # self.max_utts_evaluation_metrics = self._cfg.get('max_utts_evaluation_metrics')

        # if self.max_utts_evaluation_metrics is not None:
        #     logging.warning(
        #         'Metrics will be evaluated on first %d examples of the evaluation datasets.',
        #         self.max_utts_evaluation_metrics,
        #     )

        # # Regularization
        # self.eps = self._cfg.get('eps', 1e-8)

        # # Setup optional Optimization flags
        # self.setup_optimization_flags()

        logging.debug('Initialized              %s', self.__class__.__name__)
        logging.debug('\tdoing SSL-pretraining: %s', (self.ssl_pretrain_masking is not None))
        logging.debug('\tp_cond:                %s', self.p_cond)
        logging.debug('\tnormalize_input:       %s', self.normalize_input)
        logging.debug('\tloss:                  %s', self.loss)
        logging.debug('\teps:                   %s', self.eps)

    def _setup_train_dataloader(self, dataset_config, dataloader_params):
        dataset = create_vocoder_dataset(
            dataset_type=dataset_config.dataset_type,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=True

        )
        sampler = dataset.get_sampler(batch_size=dataloader_params.batch_size, world_size=self.trainer.world_size)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **dataloader_params
        )
        return data_loader

    def _setup_test_dataloader(self, dataset_config, dataloader_params):
        dataset = create_vocoder_dataset(
            dataset_type=dataset_config.dataset_type,
            dataset_args=dataset_config.dataset_args,
            is_train=False
        )
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_params)
        return data_loader

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            self._train_dl = self._setup_train_dataloader(
                dataset_config=train_data_config.dataset, dataloader_params=train_data_config.dataloader_params
            )
        else:
            super().setup_training_data(train_data_config)

    # def setup_validation_data(self, cfg):
    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'is_tarred' in val_data_config and val_data_config['is_tarred']:
            self._validation_dl = self._setup_test_dataloader(
                dataset_config=val_data_config.dataset, dataloader_params=val_data_config.dataloader_params
            )
        else:
            super().setup_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'is_tarred' in test_data_config and test_data_config['is_tarred']:
            self._test_dl = self._setup_test_dataloader(
                dataset_config=test_data_config.dataset, dataloader_params=test_data_config.dataloader_params
            )
        else:
            super().setup_test_data(test_data_config)

    @torch.inference_mode()
    def forward_internal(self, input_signal, input_length=None, enable_ssl_masking=False):
        """Internal forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T] or [B, T, C]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            enable_ssl_masking: Whether to enable SSL masking of the input. If using SSL pretraining, masking
                is applied to the input signal. If not using SSL pretraining, masking is not applied.

        Returns:
            Output signal `output` in the time domain and the length of the output signal `output_length`.
        """
        batch_length = input_signal.size(-1)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)

        # Encoder
        encoded, encoded_length = self.encoder(input=input_signal, input_length=input_length)

        # Initial process state
        init_state = torch.randn_like(encoded) * self.flow.sigma_start

        # Codec
        with torch.no_grad():
            input_codec, input_codec_length = self.codec(input=input_signal, input_length=input_length)
            input_codec = einops.rearrange(input_codec, 'B D T -> B 1 D T')

            # Reshape the codec output to match the encoder output
            estimator_condition = input_codec
            # easier case: estimator_condition is smaller in length and dimension
            assert estimator_condition.shape[-1] <= encoded.shape[-1]
            assert estimator_condition.shape[-2] <= encoded.shape[-2]
            
            max_length = max(encoded.shape[-1], estimator_condition.shape[-1])
            max_dim = max(encoded.shape[-2], estimator_condition.shape[-2])

            estimator_condition = torch.nn.functional.pad(
                estimator_condition, (0, max_length - estimator_condition.shape[-1], 0, max_dim - estimator_condition.shape[-2])
            )

        # Conditional input
        if self.p_cond == 0:
            # The model is trained without the conditional input
            encoded = torch.zeros_like(encoded)
        elif enable_ssl_masking and self.ssl_pretrain_masking is not None:
            # Masking for self-supervised pretraining
            encoded = self.ssl_pretrain_masking(input_spec=encoded, length=encoded_length)

        estimator_condition = torch.cat([encoded, estimator_condition], dim=1)

        # Sampler
        generated, generated_length = self.sampler(
            state=init_state, estimator_condition=estimator_condition, state_length=encoded_length
        )

        # Decoder
        output, output_length = self.decoder(input=generated, input_length=generated_length)

        if self.normalize_input:
            # rescale to the original scale
            output = output * norm_scale

        # Trim or pad the estimated signal to match input length
        output = self.match_batch_length(input=output, batch_length=batch_length)

        return output, output_length

    @typecheck(
        input_types={
            "target_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "loss": NeuralType(None, LossType()),
        },
    )
    def _step(self, target_signal, input_signal, input_length=None):
        batch_size = target_signal.size(0)

        if self.normalize_input:
            # max for each example in the batch
            norm_scale = torch.amax(input_signal.abs(), dim=(-1, -2), keepdim=True)
            # scale input signal
            input_signal = input_signal / (norm_scale + self.eps)
            # scale the target signal
            target_signal = target_signal / (norm_scale + self.eps)

        # Apply encoder to both target and the input
        input_enc, input_enc_len = self.encoder(input=input_signal, input_length=input_length)
        target_enc, _ = self.encoder(input=target_signal, input_length=input_length)

        # Self-Supervised Pretraining
        if self.ssl_pretrain_masking is not None:
            input_enc = self.ssl_pretrain_masking(input_spec=input_enc, length=input_enc_len)

        # Drop off conditional inputs (input_enc) with (1 - p_cond) probability.
        # The dropped conditions will be set to zeros
        keep_conditions = einops.rearrange((torch.rand(batch_size) < self.p_cond).float(), 'B -> B 1 1 1')
        input_enc = input_enc * keep_conditions.to(input_enc.device)

        x_start = torch.zeros_like(input_enc)

        time = self.flow.generate_time(batch_size=batch_size).to(device=input_enc.device)
        sample = self.flow.sample(time=time, x_start=x_start, x_end=target_enc)

        # The vector field model is conditioned on the input signal codec
        with torch.no_grad():
            input_codec, input_codec_len = self.codec(input=input_signal, input_length=input_length)
            input_codec = einops.rearrange(input_codec, 'B D T -> B 1 D T')

            # Reshape the codec output to match the encoder output
            estimator_condition = input_codec
            # easier case: estimator_condition is smaller in length and dimension
            assert estimator_condition.shape[-1] <= input_enc.shape[-1]
            assert estimator_condition.shape[-2] <= input_enc.shape[-2]
            
            max_length = max(input_enc.shape[-1], estimator_condition.shape[-1])
            max_dim = max(input_enc.shape[-2], estimator_condition.shape[-2])

            estimator_condition = torch.nn.functional.pad(
                estimator_condition, (0, max_length - estimator_condition.shape[-1], 0, max_dim - estimator_condition.shape[-2])
            )

        # we want to get a vector field estimate given current state
        # at training time, current state is sampled from the conditional path
        #   the vector field model is also conditioned on input signal
        estimator_input = torch.cat([sample, input_enc, estimator_condition], dim=-3)

        # Estimate the vector  using the neural estimator
        estimate, estimate_len = self.estimator(input=estimator_input, input_length=input_enc_len, condition=time)

        conditional_vector_field = self.flow.vector_field(time=time, x_start=x_start, x_end=target_enc, point=sample)

        return self.loss(estimate=estimate, target=conditional_vector_field, input_length=input_enc_len)