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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any

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
from nemo.collections.tts.parts.utils.callbacks import _get_logger, ArtifactGenerator, AudioArtifact, ImageArtifact
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback as TTSLoggingCallback
from nemo.utils import logging
from nemo.utils.decorators import experimental

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


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
        generators: List[ArtifactGenerator] = None,
        data_loader: torch.utils.data.DataLoader = None,
        log_epochs: Optional[List[int]] = None,
        epoch_frequency: int = 1,
        output_dir: Optional[Path] = None,
        loggers: Optional[List[Logger]] = None,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
    ):
        if log_tensorboard:
            try:
                _get_logger(loggers, TensorBoardLogger)
            except Exception as e:
                logging.warning(f"Could not find {TensorBoardLogger} logger in {loggers}.")
                logging.warning(f"Could not create tensorboard logger: {e}")
                log_tensorboard = False
        else:
            log_tensorboard = False

        if log_wandb:
            if not HAVE_WANDB:
                raise ValueError("Wandb not installed.")
            try:
                _get_logger(loggers, WandbLogger)
            except Exception as e:
                logging.warning(f"Could not find {WandbLogger} logger in {loggers}.")
                logging.warning(f"Could not create wandb logger: {e}")
                log_wandb = False
        else:
            log_wandb = False

        super().__init__(
            generators=None, data_loader=None, log_epochs=None, epoch_frequency=1,
            output_dir=output_dir, loggers=loggers, log_tensorboard=log_tensorboard, log_wandb=log_wandb)

    def _log_audio(self, audio: AudioArtifact, log_dir: Path, step: int):
        # Convert torch.Tensor to numpy array if needed
        audio_data = audio.data
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        # Ensure audio_data is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Handle invalid values (NaN, Inf)
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            logging.warning(f"Audio {audio.id} contains NaN or Inf values, replacing with zeros")
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct shape: squeeze to 1D if needed
        if audio_data.ndim > 1:
            # If shape is (1, N) or (N, 1), squeeze to 1D
            if audio_data.shape[0] == 1:
                audio_data = audio_data.squeeze(0)
            elif audio_data.shape[1] == 1:
                audio_data = audio_data.squeeze(1)
        
        # Ensure audio is not empty
        if audio_data.size == 0:
            logging.warning(f"Audio {audio.id} is empty, skipping logging")
            return
        
        # Clip values to valid range for float32
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        if log_dir:
            filepath = log_dir / audio.filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)
            sf.write(file=filepath, data=audio_data, samplerate=audio.sample_rate)

        if self.tensorboard_logger:
            self.tensorboard_logger.add_audio(
                tag=audio.id,
                snd_tensor=audio.data if isinstance(audio.data, torch.Tensor) else torch.from_numpy(audio_data),
                global_step=step,
                sample_rate=audio.sample_rate,
            )

        logging.info(f"Logging audio to wandb: {audio.id}")
        logging.info(f"Wandb logger: {self.wandb_logger}")
        if self.wandb_logger:
            logging.info(f"Logging audio to wandb: {audio.id}")
            wandb_audio = (wandb.Audio(audio_data, sample_rate=audio.sample_rate, caption=f"[step: {step}] {audio.id}"),)
            self.wandb_logger.log({audio.id: wandb_audio})
            logging.info(f"Wandb logged audio: {audio.id}")

    def on_fit_start(self, trainer: Trainer, model: LightningModule):
        """Log initial data artifacts."""
        if self.data_loader is None:
            logging.warning('Data loader is not set, skipping initial artifacts log.')
            return

        super().on_fit_start(trainer, model)

    def on_train_epoch_end(self, trainer: Trainer, model: LightningModule):
        """Log artifacts at the end of an epoch."""
        if self.data_loader is None:
            logging.warning('Data loader is not set, skipping epoch artifacts log.')
            return

        super().on_train_epoch_end(trainer, model)

    def on_validation_batch_end(self, trainer: Trainer, model: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Log artifacts at the end of a validation batch."""
        epoch = 1 + model.current_epoch

        audio_list = []
        image_list = []

        if len(outputs['audio_list']) == len(outputs['image_list']) == 0:
            logging.debug('List are empty, no artifacts to log at batch %d.', batch_idx)
            return

        for audio in outputs['audio_list']:
            logging.info(f"Appending audio to list")
            audio_list.append(AudioArtifact(id=audio['id'], data=audio['data'], sample_rate=audio['sample_rate'], filepath=audio['filepath']))
        for image in outputs['image_list']:
            logging.info(f"Appending image to list")
            image_list.append(ImageArtifact(id=image['id'], data=image['data'], filepath=image['filepath'], x_axis=image['x_axis'], y_axis=image['y_axis']))

        log_dir = self.output_dir / f"val_epoch_{epoch}_batch_{batch_idx}" if self.output_dir else None

        self._log_artifacts(audio_list=audio_list, image_list=image_list, log_dir=log_dir, global_step=trainer.global_step)

