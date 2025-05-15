# cd ~/personal/codes/nemo_codec_fm_decoding

export HYDRA_FULL_ERROR=1
export PYTHONPATH=.

export WANDB_API_KEY="a256061eb5311e7d96a9f735f71737154a1b9bed"
export DEBUG=true

# Check if torchaudio is installed
python3 -c "import torchaudio" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "torchaudio not installed, installing..."
    apt update && apt install -y ffmpeg sox libavdevice-dev
    bash ./scripts/installers/install_torchaudio_latest.sh
else
    echo "torchaudio is installed"
fi


# Set experiment parameters
EXP="0a"
EPOCH=400
LR="2.5e-4"
WD="1e-4"
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=32
PRECISION=16
OPTIM="adamw"
SCHE="CosineAnnealing"
MIN_LR="1e-6"

# Set experiment name
EXP_NAME=codec_flow_matching
PROJECT_NAME="codec_flow_matching"
VERSION_NAME="version_${EXP}"

# Set data paths
CV_TRAIN_MANIFEST=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/cv_audio_44khz_tar/log_manifest_only_en.json
CV_VAL_MANIFEST=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/TTS/cv_audio_44khz_tar/val_manifest_only_en.json

# Set config file paths
CONFIG_PATH="examples/audio/conf"
CONFIG_NAME="codec_fm_decoding.yaml"

if [[ $DEBUG == true ]]
then
  PYTHON_SCRIPT="scripts/codec_fm_decoder/debug_wrapper.py"
  PYTHON_SCRIPT="-m torch.distributed.run --nproc_per_node=8 ${PYTHON_SCRIPT}"
  EXP_NAME="${EXP_NAME}_debug"
  echo "Debug mode enabled. Using debug wrapper script."
else
  PYTHON_SCRIPT="examples/audio/audio_to_audio_train.py"
fi

RESUME_IF_EXISTS=true
if [[ $RESUME_IF_EXISTS == true ]]
then
  EXP_NAME="${EXP_NAME}_resume"
  echo "Resuming from previous checkpoint..."
  if [ ! -d "nemo_experiments/${EXP_NAME}/${VERSION_NAME}" ]; then
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
    ++exp_manager.version=${VERSION_NAME} \
    exp_manager.resume_if_exists=${RESUME_IF_EXISTS} \
    trainer.precision=${PRECISION} \
    trainer.max_epochs=${EPOCH} \
    trainer.devices=-1 \
    model.train_ds.manifest_filepath=${CV_TRAIN_MANIFEST} \
    model.train_ds.batch_size=${TRAIN_BATCH_SIZE} \
    model.validation_ds.manifest_filepath=${CV_VAL_MANIFEST} \
    model.validation_ds.batch_size=${EVAL_BATCH_SIZE} \
    model.optim.name=${OPTIM} \
    model.optim.lr=${LR} \
    model.optim.weight_decay=${WD} \
    model.optim.sched.name=${SCHE} \
    model.optim.sched.min_lr=${MIN_LR}
EOF

echo "Running command: $cmd"

bash -c "${cmd}"
