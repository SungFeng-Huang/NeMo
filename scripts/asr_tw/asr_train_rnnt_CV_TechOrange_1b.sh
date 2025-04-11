CV_MANIFEST=/lustre/fs12/portfolios/nvr/users/szuweif/Datasets/cv-corpus-20.0-2024-12-06/zh-TW/CommonVoice_VQScore_space.jsonl.clean_m3
CV_TRAIN_MANIFEST=/lustre/fs12/portfolios/nvr/users/szuweif/Datasets/cv-corpus-20.0-2024-12-06/zh-TW/CommonVoice_VQScore_train
CV_VAL_MANIFEST=/lustre/fs12/portfolios/nvr/users/szuweif/Datasets/cv-corpus-20.0-2024-12-06/zh-TW/CommonVoice_VQScore_valid

TO_TRAIN_MANIFEST='/datasets/TechOrange/techorange_formated_train.jsonl.clean_m3'
TO_VAL_MANIFEST='/datasets/TechOrange/techorange_formated_valid.jsonl.clean_m3'

# TOKENIZERS_DIR='/datasets/TechOrange/mount/src/NeMo/ASR/CV_TechOrange/tokenizers/tokenizer_spe_bpe_v7000/tokenizer_spe_bpe_v7000'
TOKENIZERS_DIR='/datasets/TechOrange/mount/src/NeMo/ASR/CV_TechOrange/tokenizers/tokenizer_spe_bpe_v5000/tokenizer_spe_bpe_v5000'
# TOKENIZERS_DIR='/datasets/TechOrange/mount/src/NeMo/ASR/TechOrange_tp1/tokenizers/tokenizer_spe_bpe_v5000'

PRETRAINED_MODEL='/results/checkpoints/aishell1_fc_rnnt_bpe_5000_50_n1_bs64_lr2.5e-4_a100_600M_spec00.nemo'
# PRETRAINED_MODEL='/results/checkpoints/FastConformerXL-Hybrid-Transducer-CTC-BPE-averaged.nemo'


EPOCH=400 #100 #400
LR="2.5e-4" #5 #"2e-3" #5 #"2.5e-3" #"1e-3"
WD="1e-4" # 1e-3
node=1 #8
TRAIN_BATCH_SIZE=64 #64 #32 #16
EVAL_BATCH_SIZE=32 #32 #16
FUSED_BATCH_SIZE=4
DROPOUT=0.3
PL=2
PRECISION=16
### SGD NOT BETA!
OPTIM="adamw" #"adamw"
#NOAM need d_model!
SCHE="CosineAnnealing" #"NoamAnnealing"
MIN_LR="1e-6" #1e-6


EXP_NAME=aishell1_fc_rnnt_bpe_1b_5000_${EPOCH}_n${node}_bs${TRAIN_BATCH_SIZE}_lr${LR}_a100_600M_spec00
PROJECT_NAME="aishell1_fct_asr"
VERSION_NAME="version_1b"

# WandB info
export WANDB_API_KEY="a256061eb5311e7d96a9f735f71737154a1b9bed" 

# Config file
CONFIG_PATH="/codes/examples/asr/conf/fastconformer/hybrid_transducer_ctc"
# CONFIG_NAME=model.yaml
CONFIG_NAME=model_600m_mixedtok_rnnt.yaml

# Necessary Exports
export HYDRA_FULL_ERROR=1
SLURM_JOB_NUM_GPUS=8

# Model configs
ENCODING="bpe" # char or bpe

if [[ "$ENCODING" == "char" ]]
then
  SCRIPT_POSTFIX=""
else
  SCRIPT_POSTFIX="_bpe"
fi

export DEBUG=false
export HOSTNAME=$(hostname -I | awk '{print $1}')
export DEBUG_PORT=5678
export OMP_NUM_THREADS=16
# ssh -fN -L localhost:$DEBUG_PORT:cs-oci-ord-vscode-02.nvidia.com:$DEBUG_PORT $USER@$HOSTNAME

if [[ $DEBUG == true ]]
then
  PYTHON_SCRIPT="scripts/asr_tw/debug_wrapper.py"
  PYTHON_SCRIPT="-m torch.distributed.run --nproc_per_node=${SLURM_JOB_NUM_GPUS} ${PYTHON_SCRIPT}"
  EXP_NAME="${EXP_NAME}_debug"
  echo "Debugging mode enabled. Using debug wrapper script."
else
  PYTHON_SCRIPT="examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py"
fi

RESUME_IF_EXISTS=true
if [[ $RESUME_IF_EXISTS == true ]]
then
  EXP_NAME="${EXP_NAME}_resume"
  echo "Resuming from previous checkpoint..."
  # Check if the checkpoint directory exists
  if [ ! -d "nemo_experiments/${EXP_NAME}/${VERSION_NAME}" ]; then
    echo "Checkpoint directory does not exist."
    RESUME_IF_EXISTS=false
  fi
else
  echo "Not resuming from previous checkpoint."
fi

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& wandb login ${WANDB_API_KEY} \
&& HYDRA_FULL_ERROR=1 PYTHONPATH=. python ${PYTHON_SCRIPT} \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.project=$PROJECT_NAME \
    exp_manager.wandb_logger_kwargs.name=$EXP_NAME \
    exp_manager.name=$EXP_NAME \
    ++exp_manager.version=$VERSION_NAME \
    exp_manager.resume_if_exists=$RESUME_IF_EXISTS \
    ++exp_manager.max_time_per_run="00:03:45:00" \
    trainer.log_every_n_steps=365 \
    trainer.precision=$PRECISION \
    trainer.check_val_every_n_epoch=1 \
    trainer.max_epochs=$EPOCH \
    ++trainer.devices=-1 \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    model.train_ds.pin_memory=true \
    ++model.train_ds.use_start_end_token=false \
    model.train_ds.max_duration=20.0 \
    model.train_ds.num_workers=2 \
    ++model.train_ds.shuffle_n=2048 \
    ++model.train_ds.manifest_filepath="[$TO_TRAIN_MANIFEST, $CV_TRAIN_MANIFEST]" \
    model.train_ds.batch_size=$TRAIN_BATCH_SIZE \
    +model.train_ds.use_lhotse=false \
    +model.train_ds.batch_duration=200 \
    model.validation_ds.pin_memory=true \
    ++model.validation_ds.use_start_end_token=false \
    model.validation_ds.num_workers=1 \
    model.validation_ds.manifest_filepath="[$TO_VAL_MANIFEST, $CV_VAL_MANIFEST]" \
    model.validation_ds.batch_size=$EVAL_BATCH_SIZE \
    +model.validation_ds.use_lhotse=false \
    +model.validation_ds.use_bucketing=false \
    +model.validation_ds.max_cuts=8 \
    model.joint.fused_batch_size=$FUSED_BATCH_SIZE \
    model.encoder.dropout_emb=0 \
    model.encoder.dropout_att=0.1 \
    model.encoder.dropout=0.3 \
    model.joint.jointnet.dropout=$DROPOUT \
    model.decoder.prednet.pred_rnn_layers=$PL \
    model.tokenizer.dir=$TOKENIZERS_DIR \
    model.tokenizer.type=bpe \
    model.optim.name=$OPTIM \
    model.optim.lr=$LR \
    model.optim.weight_decay=$WD \
    model.optim.sched.name=$SCHE \
    model.optim.sched.warmup_steps=10000 \
    model.optim.sched.min_lr=$MIN_LR
EOF

if [[ $RESUME_IF_EXISTS == false ]]
then
  cmd="${cmd} \
    ++init_from_nemo_model.model0.path=$PRETRAINED_MODEL "
fi
echo "cmd: $cmd"
#   ++init_from_nemo_model.model0.exclude='["decoder", "joint"]' \
#   ++init_from_nemo_model.model0.exclude='["decoder.prediction.embed.weight", "joint.joint_net.2"]' \

bash -c "${cmd}"

