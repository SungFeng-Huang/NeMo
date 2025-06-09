from typing import Any, Dict, Optional, Tuple, Union

import einops
import hydra
import numpy as np
import torch
import wandb
from wandb.wandb_run import Run
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from nemo.collections.audio.models.audio_to_audio import AudioToAudioModel
from nemo.collections.audio.models.enhancement import FlowMatchingAudioToAudioModel
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.data.vocoder_web_dataset import create_vocoder_dataset
from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
# from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.parts.utils.helpers import (
    plot_spectrogram_to_numpy,
    save_figure_to_numpy,
)
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
        self.num_groups = self.codec.vector_quantizer.num_groups
        if self.feature_type == "quantized":
            self.feature_dim = self.codec.vector_quantizer.num_groups
        elif self.feature_type in ["postquantized", "prequantized"]:
            self.feature_dim = self.codec.vector_quantizer.codebook_dim_per_group * self.codec.vector_quantizer.num_groups

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
            "input_signal": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, input_signal: torch.Tensor, input_length: Optional[torch.Tensor] = None
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
        B, T = input_signal.size(0), input_signal.size(-1)
        input_signal = input_signal.view(B, T)

        # codec output (B, C, F, N)
        self.codec.eval()
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32), torch.no_grad():
            if self.feature_type == "prequantized":
                # encoded (B, codebook_dim_per_group * num_groups, T)
                # encoded_len (B)
                encoded, encoded_len = self.codec.encode_audio(audio=input_signal, audio_len=input_length)
                return encoded, encoded_len
            
            elif self.feature_type in ["quantized", "postquantized"]:
                # FSD uses rounded scalar to determin code_id
                # gen_tokens (B, num_groups, T)
                gen_tokens, gen_tokens_lens = self.codec.encode(audio=input_signal, audio_len=input_length)

                if self.feature_type == "quantized":
                    return gen_tokens, gen_tokens_lens

                elif self.feature_type == "postquantized":
                    # get dequantized representation
                    # dequantized (B, codebook_dim_per_group * num_groups, T)
                    dequantized = self.codec.dequantize(tokens=gen_tokens, tokens_len=gen_tokens_lens)
                    return dequantized, gen_tokens_lens
                
            else:
                raise ValueError(f"Unknown feature type {self.feature_type}. Supported types are quantized, postquantized, prequantized")

    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude all parameters."""
        return {}  # Return empty state dict to exclude from checkpoint

    def load_state_dict(self, state_dict, strict=False):
        pass
        # """Override load_state_dict to handle codec parameters."""
        # # Filter out codec-related parameters
        # filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('codec.')}
        # # Load the filtered state dict
        # super().load_state_dict(filtered_state_dict, strict=False)


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
        super().__init__(cfg=cfg, trainer=trainer)

        # Codec
        self.codec = self.from_config_dict(self._cfg.codec)

        self.log_media = self._cfg.log_config.get("log_media", True)
        self.loss_masked = self._cfg.get('loss_masked', False)

        logging.debug('Initialized              %s', self.__class__.__name__)
        logging.debug('\tdoing SSL-pretraining: %s', (self.ssl_pretrain_masking is not None))
        logging.debug('\tp_cond:                %s', self.p_cond)
        logging.debug('\tnormalize_input:       %s', self.normalize_input)
        logging.debug('\tloss:                  %s', self.loss)
        logging.debug('\teps:                   %s', self.eps)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude codec parameters."""
        state_dict = super().state_dict(*args, **kwargs)
        # Remove all codec-related parameters
        keys_to_remove = [k for k in state_dict.keys() if k.startswith('codec.')]
        for k in keys_to_remove:
            del state_dict[k]
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        """Override load_state_dict to handle codec parameters."""
        # Filter out codec-related parameters
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('codec.')}
        # Load the filtered state dict
        super().load_state_dict(filtered_state_dict, strict=False)

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
        input_enc, input_enc_len = self.encoder(input=input_signal, input_length=input_length)

        # Codec
        with torch.no_grad():
            input_codec, input_codec_len = self.codec(input_signal=input_signal, input_length=input_length)
            input_codec = einops.rearrange(input_codec, 'B D T -> B 1 D T')

        estimator_condition = self._get_estimator_condition(input_enc, input_codec)

        # Conditional input
        if self.p_cond == 0:
            # The model is trained without the conditional input
            input_enc = torch.zeros_like(input_enc)
            estimator_condition = torch.zeros_like(estimator_condition)
        elif enable_ssl_masking and self.ssl_pretrain_masking is not None:
            # Masking for self-supervised pretraining
            with torch.no_grad():
                mask = torch.ones_like(input_enc)
                mask = self.ssl_pretrain_masking(input_spec=mask, length=input_enc_len)
                mask = mask.bool()
            input_enc = input_enc * mask

        # Concatenate the encoded and estimator_condition
        estimator_condition = torch.cat([input_enc, estimator_condition], dim=1)

        # Initial process state
        init_state = torch.randn_like(input_enc) * self.flow.sigma_start

        # Sampler
        generated, generated_length = self.sampler(
            state=init_state, estimator_condition=estimator_condition, state_length=input_enc_len
        )

        # Replace unmasked spectrogram generated with the encoded spectrogram
        if enable_ssl_masking and self.ssl_pretrain_masking is not None:
            generated = torch.where(mask, input_enc, generated)

        # Decoder
        output, output_length = self.decoder(input=generated, input_length=generated_length)

        if self.normalize_input:
            # rescale to the original scale
            output = output * norm_scale

        # Trim or pad the estimated signal to match input length
        output = self.match_batch_length(input=output, batch_length=batch_length)

        return output, output_length

    @torch.no_grad()
    def _get_estimator_condition(self, input_enc, input_codec):
        # easier case: estimator_condition is smaller in length and dimension
        assert input_codec.shape[-1] <= input_enc.shape[-1]
        assert input_codec.shape[-2] <= input_enc.shape[-2]

        # Reshape the codec output to match the encoder output
        input_enc_length, input_enc_dim = input_enc.shape[-1], input_enc.shape[-2]
        input_codec_length, input_codec_dim = input_codec.shape[-1], input_codec.shape[-2]

        max_length = max(input_enc_length, input_codec_length)
        max_dim = max(input_enc_dim, input_codec_dim)

        estimator_condition = torch.nn.functional.pad(
            input_codec, (0, max_length - input_codec_length, 0, max_dim - input_codec_dim)
        )
        return estimator_condition

    # @typecheck()
    def _step(self, target_signal, input_signal, input_length=None, batch_idx=None):
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

        # The vector field model is conditioned on the input signal codec
        with torch.no_grad():
            input_codec, input_codec_len = self.codec(input_signal=input_signal, input_length=input_length)
            input_codec = einops.rearrange(input_codec, 'B D T -> B 1 D T')

        # Self-Supervised Pretraining
        mask = None
        if self.ssl_pretrain_masking is not None:
            with torch.no_grad():
                mask = torch.ones_like(input_enc)
                mask = self.ssl_pretrain_masking(input_spec=mask, length=input_enc_len)
                mask = mask.bool()
            input_enc = input_enc * mask

        estimator_condition = self._get_estimator_condition(input_enc, input_codec)

        # Drop off conditional inputs (input_enc) with (1 - p_cond) probability.
        # The dropped conditions will be set to zeros
        keep_conditions = einops.rearrange((torch.rand(batch_size) < self.p_cond).float(), 'B -> B 1 1 1')
        input_enc = input_enc * keep_conditions.to(input_enc.device)
        estimator_condition = estimator_condition * keep_conditions.to(estimator_condition.device)
        # Drop off the input_enc with (1 - p_cond) probability.
        keep_conditions = einops.rearrange((torch.rand(batch_size) < self.p_cond).float(), 'B -> B 1 1 1')
        input_enc = input_enc * keep_conditions.to(input_enc.device)

        # Sample from the flow
        x_start = torch.zeros_like(input_enc)
        time = self.flow.generate_time(batch_size=batch_size).to(device=input_enc.device)
        sample = self.flow.sample(time=time, x_start=x_start, x_end=target_enc)

        # we want to get a vector field estimate given current state
        # at training time, current state is sampled from the conditional path
        #   the vector field model is also conditioned on input signal
        estimator_input = torch.cat([sample, input_enc, estimator_condition], dim=-3)

        # Estimate the vector field using the neural estimator
        estimate, estimate_len = self.estimator(input=estimator_input, input_length=input_enc_len, condition=time)

        conditional_vector_field = self.flow.vector_field(time=time, x_start=x_start, x_end=target_enc, point=sample)

        # Calculate the loss
        loss = None
        if self.loss_masked and self.ssl_pretrain_masking is not None:
            length_mask = make_seq_mask_like(lengths=input_enc_len, like=input_enc, time_dim=-1, valid_ones=True)
            length_mask = length_mask.expand_as(input_enc).bool()
            loss_mask = ~mask.bool() * length_mask
            loss = self.loss(estimate=estimate, target=conditional_vector_field, mask=loss_mask)

        else:
            loss = self.loss(estimate=estimate, target=conditional_vector_field, input_length=input_enc_len)

        # Log the results
        output = {
            'loss': loss,
            'estimate': estimate.detach().cpu(),
            'conditional_vector_field': conditional_vector_field.detach().cpu(),
            'input_enc': input_enc.detach().cpu(),
            'input_codec': input_codec.detach().cpu(),
            'estimator_condition': estimator_condition.detach().cpu(),
            'sample': sample.detach().cpu(),
            'target_enc': target_enc.detach().cpu(),
            # 'time': time.detach().cpu(),
            # 'x_start': x_start.detach().cpu(),
            # 'input_signal': input_signal.detach().cpu(),
            # 'input_length': input_length.detach().cpu(),
            # 'target_signal': target_signal.detach().cpu(),
        }

        return output

    @rank_zero_only
    def log_image(self, key: str, image: Any, step: Optional[int] = None, **kwargs: Any) -> None:
        r"""Log images (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the image files
            image: The image file path, or numpy array to be logged
            step: The step number to be used for logging the image files
            \**kwargs: Optional kwargs are lists passed to each ``Wandb.Image`` instance (ex: caption, sample_rate).

        Optional kwargs are lists passed to each image (ex: caption, sample_rate).

        """
        image = plot_spectrogram_to_numpy(np.abs(image.detach().cpu().numpy()))
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_writer = logger.experiment
                tb_writer.add_image(key, image, step, dataformats="HWC")
            
            elif isinstance(logger, WandbLogger):
                if not hasattr(self, "wandb_metrics"):
                    self.wandb_metrics = {}

                wandb_logger: Run = logger.experiment
                kwargs["caption"] = f"step: {step}"
                for k in kwargs:
                    kwargs[k] = [kwargs[k]]

                n = len([image])
                kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
                metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip([image], kwarg_list)]}
                # logger.log_metrics(metrics, step=step)  # type: ignore[arg-type]
                self.wandb_metrics.update(metrics)

    def save_spectrogram_to_image(self, key: str, spectrogram: Any, step: Optional[int] = None, **kwargs: Any) -> None:
        import matplotlib.image
        matplotlib.image.imsave(f"{self.trainer.log_dir}/{key}.png", plot_spectrogram_to_numpy(np.abs(spectrogram[0,0].detach().cpu().numpy())))

    @rank_zero_only
    def log_audio(self, key: str, audio: Any, step: Optional[int] = None, **kwargs: Any) -> None:
        r"""Log audios (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the audio files
            audio: The audio file path, or numpy array to be logged
            step: The step number to be used for logging the audio files
            \**kwargs: Optional kwargs are lists passed to each ``Wandb.Audio`` instance (ex: caption, sample_rate).

        Optional kwargs are lists passed to each audio (ex: caption, sample_rate).

        """
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_writer = logger.experiment
                tb_writer.add_audio(key, audio, step, **kwargs)
            
            elif isinstance(logger, WandbLogger):
                if not hasattr(self, "wandb_metrics"):
                    self.wandb_metrics = {}

                wandb_logger: Run = logger.experiment
                audios = [audio]
                kwargs["caption"] = f"step: {step}"
                for k in kwargs:
                    kwargs[k] = [kwargs[k]]

                n = len(audios)
                kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

                metrics = {key: [wandb.Audio(audio, **kwarg) for audio, kwarg in zip(audios, kwarg_list)]}
                # logger.log_metrics(metrics, step=step)  # type: ignore[arg-type]
                self.wandb_metrics.update(metrics)

    @rank_zero_only
    def log_commit(self, step):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger: Run = logger.experiment
                # wandb_logger.log({}, commit=True)
                if not hasattr(self, "wandb_metrics"):
                    self.wandb_metrics = {}
                elif len(self.wandb_metrics) > 0:
                    logger.log_metrics(self.wandb_metrics, step=step)
                    self.wandb_metrics = {}

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch.get('target_signal', input_signal.clone())
        else:
            input_signal, input_length, target_signal, _ = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, "B T -> B 1 T")
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, "B T -> B 1 T")

        # Calculate the loss
        output = self._step(target_signal=target_signal, input_signal=input_signal, input_length=input_length, batch_idx=batch_idx)
        loss = output['loss']

        # Logs
        self.log('train_loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        if self.log_media:
            plot_id = 0
            log_media = (self.trainer.global_step + 1) % (self.trainer.log_every_n_steps * 20) == 0
            if log_media:
                for key, value in output.items():
                    if key != 'loss':
                        self.log_image(key=f"train_{plot_id}_{key}", image=value[plot_id, 0], step=self.trainer.global_step+1)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_commit(step=self.trainer.global_step)

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):

        if isinstance(batch, dict):
            # lhotse batches are dictionaries
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch.get('target_signal', input_signal.clone())
            target_length = batch.get('target_length', input_length.clone())
        else:
            input_signal, input_length, target_signal, target_length = batch

        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Calculate loss
        output = self._step(
            target_signal=target_signal,
            input_signal=input_signal,
            input_length=input_length,
            batch_idx=batch_idx,
        )
        loss = output['loss']

        # Update metrics
        update_metrics = False
        if self.max_utts_evaluation_metrics is None:
            # Always update if max is not configured
            update_metrics = True
            # Number of examples to process
            num_examples = input_signal.size(0)  # batch size
        else:
            # Check how many examples have been used for metric calculation
            first_metric_name = next(iter(self.metrics[tag][dataloader_idx]))
            num_examples_evaluated = self.metrics[tag][dataloader_idx][first_metric_name].num_examples
            # Update metrics if some examples were not processed
            update_metrics = num_examples_evaluated < self.max_utts_evaluation_metrics
            # Number of examples to process
            num_examples = min(self.max_utts_evaluation_metrics - num_examples_evaluated, input_signal.size(0))

        if update_metrics:
            # Generate output signal
            output_signal, output_length = self.forward_eval(
                input_signal=input_signal[:num_examples, ...], input_length=input_length[:num_examples]
            )

            # Update metrics
            if hasattr(self, 'metrics') and tag in self.metrics:
                # Update metrics for this (tag, dataloader_idx)
                for name, metric in self.metrics[tag][dataloader_idx].items():
                    metric.update(
                        preds=output_signal,
                        target=target_signal[:num_examples, ...],
                        input_length=input_length[:num_examples],
                    )
                    if self.log_media and batch_idx == 0:
                        for plot_id in range(num_examples): 
                            self.log_audio(key=f"{tag}_{plot_id}_output_signal", audio=output_signal[plot_id, 0, :output_length[plot_id]].cpu().numpy(), step=self.trainer.global_step, sample_rate=self._validation_dl.dataset.sample_rate)
                            self.log_audio(key=f"{tag}_{plot_id}_target_signal", audio=target_signal[plot_id, 0, :input_length[plot_id]].cpu().numpy(), step=self.trainer.global_step, sample_rate=self._validation_dl.dataset.sample_rate)
                            self.log_commit(step=self.trainer.global_step)

        # Log global step
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return {f'{tag}_loss': loss}