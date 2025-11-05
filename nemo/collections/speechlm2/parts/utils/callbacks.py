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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import librosa
import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor

from nemo.collections.tts.parts.utils.helpers import create_plot
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback as TTSLoggingCallback
from nemo.utils import logging
from nemo.utils.decorators import experimental

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


def _get_logger(loggers: List[Logger], logger_type: Type[Logger]):
    for logger in loggers:
        if isinstance(logger, logger_type):
            if hasattr(logger, "experiment"):
                return logger.experiment
            else:
                return logger
    logging.warning(f"Could not find {logger_type} logger in {loggers}.")
    return None


def _load_vocoder(model_name: Optional[str], checkpoint_path: Optional[str], type: str):
    assert (model_name is None) != (
        checkpoint_path is None
    ), f"Must provide exactly one of vocoder model_name or checkpoint: ({model_name}, {checkpoint_path})"

    checkpoint_path = str(checkpoint_path)
    if type == "hifigan":
        from nemo.collections.tts.models import HifiGanModel

        model_type = HifiGanModel
    elif type == "univnet":
        from nemo.collections.tts.models import UnivNetModel

        model_type = UnivNetModel
    else:
        raise ValueError(f"Unknown vocoder type '{type}'")

    if model_name is not None:
        vocoder = model_type.from_pretrained(model_name)
    elif checkpoint_path.endswith(".nemo"):
        vocoder = model_type.restore_from(checkpoint_path)
    else:
        vocoder = model_type.load_from_checkpoint(checkpoint_path)

    return vocoder.eval()


@dataclass
class AudioArtifact:
    id: str
    data: np.ndarray
    sample_rate: int
    filepath: Path


@dataclass
class ImageArtifact:
    id: str
    data: np.ndarray
    filepath: Path
    x_axis: str
    y_axis: str


@dataclass
class LogAudioParams:
    vocoder_type: str
    vocoder_name: str
    vocoder_checkpoint_path: str
    log_audio_gta: bool = False


def create_id(filepath: Path) -> str:
    path_prefix = str(filepath.with_suffix(""))
    file_id = path_prefix.replace(os.sep, "_")
    return file_id


class ArtifactGenerator(ABC):
    @abstractmethod
    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:
        """
        Create artifacts for the input model and test batch.

        Args:
            model: Model instance being trained to use for inference.
            batch_dict: Test batch to generate artifacts for.
            initial_log: Flag to denote if this is the initial log, can
                         be used to save ground-truth data only once.

        Returns:
            List of audio and image artifacts to log.
        """


@experimental
class LoggingCallback(TTSLoggingCallback):
    """
    Callback which can log artifacts (eg. model predictions, graphs) to local disk, Tensorboard, and/or WandB.

    Args:
        generators: List of generators to create and log artifacts from.
        data_loader: Data to log artifacts for.
        log_epochs: Optional list of specific training epoch numbers to log artifacts for.
        epoch_frequency: Frequency with which to log
        output_dir: Optional local directory. If provided, artifacts will be saved in output_dir.
        loggers: Optional list of loggers to use if logging to tensorboard or wandb.
        log_tensorboard: Whether to log artifacts to tensorboard.
        log_wandb: Whether to log artifacts to WandB.
    """

    def __init__(
        self,
        generators: List[ArtifactGenerator],
        data_loader: torch.utils.data.DataLoader = None,
        log_epochs: Optional[List[int]] = None,
        epoch_frequency: int = 1,
        output_dir: Optional[Path] = None,
        loggers: Optional[List[Logger]] = None,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
    ):
        self.generators = generators
        self.data_loader = data_loader
        self.log_epochs = log_epochs if log_epochs else []
        self.epoch_frequency = epoch_frequency
        self.output_dir = Path(output_dir) if output_dir else None
        self.loggers = loggers if loggers else []
        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb

        if log_tensorboard:
            logging.info('Creating tensorboard logger')
            self.tensorboard_logger = _get_logger(self.loggers, TensorBoardLogger)
        else:
            logging.debug('Not using tensorbord logger')
            self.tensorboard_logger = None

        if log_wandb:
            if not HAVE_WANDB:
                raise ValueError("Wandb not installed.")
            logging.info('Creating wandb logger')
            self.wandb_logger = _get_logger(self.loggers, WandbLogger)
        else:
            logging.debug('Not using wandb logger')
            self.wandb_logger = None

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tlog_epochs:      %s', self.log_epochs)
        logging.debug('\tepoch_frequency: %s', self.epoch_frequency)
        logging.debug('\toutput_dir:      %s', self.output_dir)
        logging.debug('\tlog_tensorboard: %s', self.log_tensorboard)
        logging.debug('\tlog_wandb:       %s', self.log_wandb)

    def _log_audio(self, audio: AudioArtifact, log_dir: Path, step: int):
        if log_dir:
            filepath = log_dir / audio.filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)
            sf.write(file=filepath, data=audio.data, samplerate=audio.sample_rate)

        if self.tensorboard_logger:
            self.tensorboard_logger.add_audio(
                tag=audio.id,
                snd_tensor=audio.data,
                global_step=step,
                sample_rate=audio.sample_rate,
            )

        if self.wandb_logger:
            wandb_audio = (wandb.Audio(audio.data, sample_rate=audio.sample_rate, caption=audio.id),)
            self.wandb_logger.log({audio.id: wandb_audio})

    def _log_image(self, image: ImageArtifact, log_dir: Path, step: int):
        if log_dir:
            filepath = log_dir / image.filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            filepath = None

        image_plot = create_plot(output_filepath=filepath, data=image.data, x_axis=image.x_axis, y_axis=image.y_axis)

        if self.tensorboard_logger:
            self.tensorboard_logger.add_image(
                tag=image.id,
                img_tensor=image_plot,
                global_step=step,
                dataformats="HWC",
            )

        if self.wandb_logger:
            wandb_image = (wandb.Image(image_plot, caption=image.id),)
            self.wandb_logger.log({image.id: wandb_image})

    def _log_artifacts(self, audio_list: list, image_list: list, log_dir: Optional[Path] = None, global_step: int = 0):
        """Log audio and image artifacts."""
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        for audio in audio_list:
            self._log_audio(audio=audio, log_dir=log_dir, step=global_step)

        for image in image_list:
            self._log_image(image=image, log_dir=log_dir, step=global_step)

    def on_fit_start(self, trainer: Trainer, model: LightningModule):
        """Log initial data artifacts."""
        if self.data_loader is None:
            logging.warning('Data loader is not set, skipping initial artifacts log.')
            return

        audio_list = []
        image_list = []
        for batch_dict in self.data_loader:
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(model.device)

            for generator in self.generators:
                audio, images = generator.generate_artifacts(model=model, batch_dict=batch_dict, initial_log=True)
                audio_list += audio
                image_list += images

        if len(audio_list) == len(image_list) == 0:
            logging.debug('List are empty, no initial artifacts to log.')
            return

        log_dir = self.output_dir / "initial" if self.output_dir else None

        self._log_artifacts(audio_list=audio_list, image_list=image_list, log_dir=log_dir)

    def on_train_epoch_end(self, trainer: Trainer, model: LightningModule):
        """Log artifacts at the end of an epoch."""
        if self.data_loader is None:
            logging.warning('Data loader is not set, skipping epoch artifacts log.')
            return

        epoch = 1 + model.current_epoch
        if (epoch not in self.log_epochs) and (epoch % self.epoch_frequency != 0):
            return

        audio_list = []
        image_list = []
        for batch_dict in self.data_loader:
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(model.device)

            for generator in self.generators:
                audio, images = generator.generate_artifacts(model=model, batch_dict=batch_dict)
                audio_list += audio
                image_list += images

        if len(audio_list) == len(image_list) == 0:
            logging.debug('List are empty, no artifacts to log at epoch %d.', epoch)
            return

        log_dir = self.output_dir / f"epoch_{epoch}" if self.output_dir else None

        self._log_artifacts(audio_list=audio_list, image_list=image_list, log_dir=log_dir)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Log artifacts at the end of a validation batch."""
        epoch = 1 + model.current_epoch

        audio_list = outputs['audio_list']
        image_list = outputs['image_list']

        if len(audio_list) == len(image_list) == 0:
            logging.debug('List are empty, no artifacts to log at batch %d.', batch_idx)
            return

        log_dir = self.output_dir / f"val_epoch_{epoch}_batch_{batch_idx}" if self.output_dir else None

        self._log_artifacts(audio_list=audio_list, image_list=image_list, log_dir=log_dir)

