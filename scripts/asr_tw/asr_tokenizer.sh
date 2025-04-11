CV_MANIFEST='/lustre/fs12/portfolios/nvr/users/szuweif/Datasets/cv-corpus-20.0-2024-12-06/zh-TW/CommonVoice_VQScore_space.jsonl.clean_m3'
TRAIN_MANIFEST='/datasets/TechOrange/techorange_formated_train.jsonl.clean_m3'
VAL_MANIFEST='/datasets/TechOrange/techorange_formated_valid.jsonl.clean_m3'
TEST_MANIFEST='/datasets/TechOrange/techorange_formated_test.jsonl.clean_m3'

TechOrange_MANIFEST='/datasets/TechOrange/techorange_formated_all.jsonl.clean_m3'
CV_TechOrange_MANIFEST='/datasets/TechOrange/cv_techorange_formated_all.jsonl.clean_m3'

# cat $TRAIN_MANIFEST $VAL_MANIFEST $TEST_MANIFEST > $TechOrange_MANIFEST
# cat $TechOrange_MANIFEST $CV_MANIFEST > $CV_TechOrange_MANIFEST

NUM_TOKENS=5000
OUT_TOKENIZER_DIR="/datasets/TechOrange/mount/src/NeMo/ASR/CV_TechOrange/tokenizers/tokenizer_spe_bpe_v${NUM_TOKENS}"

PYTHONPATH=. python scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest=$CV_TechOrange_MANIFEST \
    --vocab_size=$NUM_TOKENS \
    --data_root=$OUT_TOKENIZER_DIR \
    --tokenizer="spe" \
    --spe_type=bpe # or "word"