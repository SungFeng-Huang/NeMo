TRAIN_MANIFEST='/mount/src/NeMo/ASR/TechOrange_tp1/mixedtok_manifests/techorange_segmented_mixed_train.jsonl'
VAL_MANIFEST='/mount/src/NeMo/ASR/TechOrange_tp1/mixedtok_manifests/techorange_segmented_mixed_valid.jsonl'
TOKENIZERS_DIR='/mount/src/NeMo/ASR/TechOrange_tp1/tokenizers/tokenizer_spe_bpe_v5000/'
PRETRAINED_MODEL='/mount/src/NeMo/ASR/TechOrange/pretrained_model/stt_enzh_600m.nemo'

EPOCH=50 #100 #400
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

EXP_NAME=aishell1_fc_rnnt_bpe_5000_${EPOCH}_n${node}_bs${TRAIN_BATCH_SIZE}_lr${LR}_a100_600M_spec00
PROJECT_NAME="aishell1_fct_asr"

# WandB info
WANDB="" 

# Config file
CONFIG_PATH="../conf/fastconformer/"
# CONFIG_NAME=model.yaml
CONFIG_NAME=model_600m_mixedtok_rnnt.yaml

# Necessary Exports
export HYDRA_FULL_ERROR=1

# Model configs
ENCODING="bpe" # char or bpe

if [[ "$ENCODING" == "char" ]]
then
  SCRIPT_POSTFIX=""
else
  SCRIPT_POSTFIX="_bpe"
fi

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& HYDRA_FULL_ERROR=1 python speech_to_text_rnnt_bpe.py  \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.project=$PROJECT_NAME \
    exp_manager.wandb_logger_kwargs.name=$EXP_NAME \
    exp_manager.name=$EXP_NAME \
    trainer.log_every_n_steps=365 \
    trainer.precision=$PRECISION \
    model.train_ds.pin_memory=true \
    model.validation_ds.pin_memory=true \
    ++model.train_ds.use_start_end_token=false \
    ++model.validation_ds.use_start_end_token=false \
    model.train_ds.max_duration=20.0 \
    model.train_ds.num_workers=2 \
    model.validation_ds.num_workers=1 \
    trainer.check_val_every_n_epoch=1 \
    trainer.max_epochs=$EPOCH \
    ++model.train_ds.shuffle_n=2048 \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.train_ds.batch_size=$TRAIN_BATCH_SIZE \
    model.validation_ds.batch_size=$EVAL_BATCH_SIZE \
    model.joint.fused_batch_size=$FUSED_BATCH_SIZE \
    model.encoder.dropout_emb=0 \
    model.encoder.dropout_att=0.1 \
    model.encoder.dropout=0.3 \
    model.joint.jointnet.dropout=$DROPOUT \
    model.decoder.prednet.pred_rnn_layers=$PL \
    model.optim.name=$OPTIM \
    model.optim.lr=$LR \
    model.optim.weight_decay=$WD \
    model.optim.sched.name=$SCHE \
    model.optim.sched.warmup_steps=10000 \
    model.tokenizer.dir=$TOKENIZERS_DIR \
    model.tokenizer.type=bpe \
    +model.validation_ds.use_lhotse=false \
    +model.validation_ds.use_bucketing=false \
    +model.validation_ds.max_cuts=8 \
    +model.train_ds.use_lhotse=false \
    +model.train_ds.batch_duration=200 \
    ++init_from_nemo_model.model0.path=$PRETRAINED_MODEL \
    ++init_from_nemo_model.model0.exclude=['decoder'] \
    model.optim.sched.min_lr=$MIN_LR
EOF

bash -c "${cmd}"

