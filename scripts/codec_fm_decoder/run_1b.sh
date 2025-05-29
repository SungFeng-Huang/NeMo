# cd ~/personal/codes/nemo_codec_fm_decoding

export HYDRA_FULL_ERROR=1
export PYDEVD_DISABLE_FILE_VALIDATION=1
export PYTHONPATH=.

export WANDB_API_KEY="a256061eb5311e7d96a9f735f71737154a1b9bed"
export WANDB_DISABLE_SYSTEM_MONITORING=true
export DEBUG=${1:-$([ "$SLURM_JOB_NAME" == "interactive" ] && echo "true" || echo "false")}

# Check if torchaudio is installed
python3 -c "import torchaudio" > /dev/null 2>&1
# Check if previous command (python3 import torchaudio) returned non-zero exit code (i.e. failed)
if [ $? -ne 0 ]; then
    echo "torchaudio not installed, installing..."
    apt update && apt install -y ffmpeg sox libavdevice-dev
    bash ./scripts/installers/install_torchaudio_latest.sh
else
    echo "torchaudio is installed"
fi


# Set experiment parameters
EXP="1b"
EPOCH=400
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=8
STEPS_PER_EPOCH=2500

# Set experiment name
EXP_NAME="${EXP}"
PROJECT_NAME="codec_flow_matching"
VERSION_NAME="version_${EXP}"

# Set data paths
CV_TRAIN_MANIFEST=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/cv_audio_44khz_tar/1619858/train_manifest_only_en.json
CV_TRAIN_TAR_PATH='"/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/cv_audio_44khz_tar/1619858/audio_\{0..199\}.tar"'
CV_VAL_MANIFEST=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/cv_audio_44khz_tar/val_manifest_only_en.json
CV_VAL_AUDIO_DIR=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/cv_audio_44khz_tar/audio_44khz
MLS_ENGLISH_TRAIN_MANIFEST=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/mls_english_audio_44khz_tar/1616011/train_manifest.json
MLS_ENGLISH_TRAIN_TAR_PATH='"/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/mls_english_audio_44khz_tar/1616011/audio_\{0..1699\}.tar"'
MLS_ENGLISH_VAL_MANIFEST=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/mls_english_audio_44khz_tar/val_manifest_unseen.json
MLS_ENGLISH_VAL_AUDIO_DIR=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/mls_english_audio_44khz_tar/audio_44khz

# Set config file paths
CONFIG_PATH="${PWD}/examples/audio/conf"
CONFIG_NAME="codec_fm_decoding"
PYTHON_SCRIPT="examples/audio/audio_to_audio_train.py"

if [[ $DEBUG == true ]]
then
  export CUDA_VISIBLE_DEVICES=0
  export PYTHON_LOG_LEVEL=DEBUG
  # PYTHON_SCRIPT="scripts/codec_fm_decoder/debug_wrapper.py"
  # PYTHON_SCRIPT="-m torch.distributed.run --nproc_per_node=8 ${PYTHON_SCRIPT}"
  EXP_NAME="${EXP_NAME}_debug"
  PROJECT_NAME="${PROJECT_NAME}_debug"
  VERSION_NAME="version_${EXP}_$(date +%Y%m%d_%H%M%S)"
  STEPS_PER_EPOCH=100
  echo "Debug mode enabled. Using debug wrapper script."
else
  export PYTHON_LOG_LEVEL=INFO
  echo "Normal mode enabled."
  PYTHON_SCRIPT="examples/audio/audio_to_audio_train.py"
fi

RESUME_IF_EXISTS=true
if [[ $RESUME_IF_EXISTS == true ]]
then
  EXP_NAME="${EXP_NAME}_resume"
  echo "Resuming from previous checkpoint..."
  if [ ! -d "nemo_experiments/${EXP_NAME}/${VERSION_NAME}/checkpoints" ]; then
    echo "Checkpoint directory does not exist."
    RESUME_IF_EXISTS=false
  fi
else
  echo "Not resuming from previous checkpoint."
fi

read -r -d '' cmd <<EOF
echo "*******Starting********" \
&& echo "---------------" \
&& nvidia-smi \
&& wandb login ${WANDB_API_KEY} \
&& HYDRA_FULL_ERROR=1 PYTHONPATH=. python ${PYTHON_SCRIPT} \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    +exp_manager.version=${VERSION_NAME} \
    exp_manager.resume_if_exists=${RESUME_IF_EXISTS} \
    trainer.max_epochs=${EPOCH} \
    trainer.devices=-1 \
    model.train_ds.dataloader_params.batch_size=${TRAIN_BATCH_SIZE} \
    model.validation_ds.dataloader_params.batch_size=${EVAL_BATCH_SIZE} \
    ++model.train_ds.dataset.dataset_args.sample_args.steps_per_epoch=${STEPS_PER_EPOCH} \
    ++model.train_ds.dataset.dataset_args.dataset_meta.cv.manifest_path=${CV_TRAIN_MANIFEST} \
    ++model.train_ds.dataset.dataset_args.dataset_meta.cv.tar_filepath=${CV_TRAIN_TAR_PATH} \
    ++model.validation_ds.dataset.dataset_args.dataset_meta.cv.manifest_path=${CV_VAL_MANIFEST} \
    ++model.validation_ds.dataset.dataset_args.dataset_meta.cv.audio_dir=${CV_VAL_AUDIO_DIR} \
    ++model.validation_ds.dataset.dataset_args.dataset_meta.mls_english={} \
    ++model.validation_ds.dataset.dataset_args.dataset_meta.mls_english.manifest_path=${MLS_ENGLISH_VAL_MANIFEST} \
    ++model.validation_ds.dataset.dataset_args.dataset_meta.mls_english.audio_dir=${MLS_ENGLISH_VAL_AUDIO_DIR} \
    ++model.train_ds.dataset.dataset_args.dataset_meta.mls_english={} \
    ++model.train_ds.dataset.dataset_args.dataset_meta.mls_english.manifest_path=${MLS_ENGLISH_TRAIN_MANIFEST} \
    ++model.train_ds.dataset.dataset_args.dataset_meta.mls_english.tar_filepath=${MLS_ENGLISH_TRAIN_TAR_PATH} \
    ++model.train_ds.dataset.dataset_args.dataset_meta.mls_english.sample_weight=3.0
EOF

# echo "Running command: $cmd"
# if [[ $DEBUG == true ]]; then
#   while true; do
#     bash -c "${cmd}"
#     echo "Training completed. Press Ctrl+C to stop or Enter to run again..."
#     read -t 10 || true # Wait for 10 seconds for input, continue if timeout
#   done
# else
#   bash -c "${cmd}"
# fi
    # ++model.train_ds.dataset.dataset_args.sample_args.dataset_weights=[0.25,0.75]

bash -c "${cmd}"