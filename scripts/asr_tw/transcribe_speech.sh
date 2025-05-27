SUBSET="test"
# DATASET_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/cv_techorange_formated_${SUBSET}.jsonl.clean_m3"
# WER_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/version_2c/cv_to_transcribed_techorange_formated_${SUBSET}.jsonl"
# VQ_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/cv_techorange_formated_${SUBSET}.jsonl.vq"
# PLOT_DIR="nemo_experiments/plots/version_2c/CV_TechOrange"

# DATASET_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/techorange_formated_${SUBSET}.jsonl.clean_m3"
# WER_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/version_2c/transcribed_techorange_formated_${SUBSET}.jsonl"
# VQ_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/VQscore/TechOrange_VQScore_${SUBSET}"
# PLOT_DIR="nemo_experiments/plots/version_2c/TechOrange"

# DATASET_MANIFEST="/lustre/fs12/portfolios/nvr/users/szuweif/Datasets/cv-corpus-20.0-2024-12-06/zh-TW/CommonVoice_VQScore_${SUBSET}"
# WER_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/version_2c/transcribed_commonvoice_${SUBSET}.jsonl"
# VQ_MANIFEST=$DATASET_MANIFEST
# PLOT_DIR="nemo_experiments/plots/version_2c/CV"

# MODEL_PATH="/home/sungfengh/personal/codes/nemo_zh-tw_asr/nemo_experiments/aishell1_fc_rnnt_bpe_2c_7000_400_n1_bs64_lr2.5e-4_a100_600M_spec00_resume/version_2c/checkpoints/aishell1_fc_rnnt_bpe_2c_7000_400_n1_bs64_lr2.5e-4_a100_600M_spec00_resume.nemo"
# TRANSCRIBE_DIR="/home/sungfengh/personal/codes/nemo_zh-tw_asr/nemo_experiments/transcribed/version_2c"
MODEL_PATH="/home/sungfengh/personal/codes/nemo_zh-tw_asr/nemo_experiments/ctc_only_version_2c/ctc-only-FastConformerXL-Hybrid-Transducer-CTC-BPE-averaged.nemo"
TRANSCRIBE_DIR="/home/sungfengh/personal/codes/nemo_zh-tw_asr/nemo_experiments/transcribed/version_2c_ctc"

# DATASET_MANIFEST="/home/sungfengh/personal/datasets/TechOrange/techorange_formated_${SUBSET}.jsonl.clean_m3"
# WER_MANIFEST="${TRANSCRIBE_DIR}/transcribed_techorange_formated_${SUBSET}.jsonl"

DATASET_MANIFEST="/lustre/fs12/portfolios/nvr/users/szuweif/Datasets/cv-corpus-20.0-2024-12-06/zh-TW/CommonVoice_VQScore_${SUBSET}"
WER_MANIFEST="${TRANSCRIBE_DIR}/transcribed_commonvoice_${SUBSET}.jsonl"

# DATASET_MANIFEST="/home/sungfengh/personal/datasets/ytr_to_audio/all_spaced.jsonl"
# WER_MANIFEST="${TRANSCRIBE_DIR}/transcribed_ytr_to_audio.jsonl"

# DATASET_MANIFEST="/home/sungfengh/personal/datasets/COSPRO/all.jsonl"
# WER_MANIFEST="${TRANSCRIBE_DIR}/transcribed_COSPRO_all.jsonl"

export HYDRA_FULL_ERROR=1
export PYTHONPATH=.

DEBUG=false
if [ "$DEBUG" == true ]; then
    SCRIPT_FILE="scripts/asr_tw/debug_transcribe.py"
    echo "Debug mode: $SCRIPT_FILE"
else
    SCRIPT_FILE="examples/asr/transcribe_speech.py"
    echo "Normal mode: $SCRIPT_FILE"
fi

TRANSCRIBE=true
if [ $TRANSCRIBE == true ]; then
    # python -m torch.distributed.run --nproc_per_node=8 s$SCRIPT_FILE \
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 $SCRIPT_FILE \
        model_path=$MODEL_PATH  \
        dataset_manifest=$DATASET_MANIFEST \
        output_filename=$WER_MANIFEST \
        channel_selector=average \
        batch_size=32 \
        preserve_alignment=True \
        preserve_attention=False \
        compute_timestamps=True \
        compute_langs=False \
        amp=True \
        append_pred=False  || { echo "Failed to transcribe"; exit 1; }
fi

PLOT=false
if [ $PLOT == true ]; then
    VQ_MANIFEST=$DATASET_MANIFEST
    PLOT_DIR="nemo_experiments/plots/version_2c/CV"
    python scripts/asr_tw/plot_relationship.py --subset $SUBSET --jsonl_file_path $WER_MANIFEST --output_dir $PLOT_DIR
    python scripts/asr_tw/plot_relationship.py --subset $SUBSET --score_type vq --vq_file_path $VQ_MANIFEST --jsonl_file_path $WER_MANIFEST --output_dir ${PLOT_DIR}_vq
    python scripts/asr_tw/plot_relationship.py --subset $SUBSET --score_type encoder_True --jsonl_file_path $WER_MANIFEST --output_dir ${PLOT_DIR}_encoder_True
    python scripts/asr_tw/plot_relationship.py --subset $SUBSET --score_type encoder_False --jsonl_file_path $WER_MANIFEST --output_dir ${PLOT_DIR}_encoder_False
    python scripts/asr_tw/plot_relationship.py --subset $SUBSET --score_type decoder_True --jsonl_file_path $WER_MANIFEST --output_dir ${PLOT_DIR}_decoder_True
    python scripts/asr_tw/plot_relationship.py --subset $SUBSET --score_type decoder_False --jsonl_file_path $WER_MANIFEST --output_dir ${PLOT_DIR}_decoder_False
fi
