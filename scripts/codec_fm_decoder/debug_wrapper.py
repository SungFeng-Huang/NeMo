import os
import debugpy
import torch

from examples.audio.audio_to_audio_train import main
from scripts.codec_fm_decoder.debug_setup import setup_debugging


# 在訓練腳本中調用
if __name__ == "__main__":
    # Set up debugpy
    setup_debugging()

    main()