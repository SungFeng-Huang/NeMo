wandb login a256061eb5311e7d96a9f735f71737154a1b9bed\
&& HYDRA_FULL_ERROR=1 PYTHONPATH=/codes python -m torch.distributed.run --nproc_per_node=8 /codes/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=/codes/examples/asr/conf/fastconformer/hybrid_transducer_ctc \
    --config-name=model_600m_mixedtok_rnnt.yaml \
    model.train_ds.manifest_filepath=/datasets/TechOrange/techorange_formated_train.jsonl.clean_m3 \
    model.validation_ds.manifest_filepath=/datasets/TechOrange/techorange_formated_valid.jsonl.clean_m3 \
    model.test_ds.manifest_filepath=/datasets/TechOrange/techorange_formated_test.jsonl.clean_m3 \
    model.tokenizer.dir=/datasets/TechOrange/mount/src/NeMo/ASR/TechOrange_tp1/tokenizers/tokenizer_spe_bpe_v5000 \
    ++exp_manager.exp_dir=/results/ \
    ++init_from_nemo_model=/results/checkpoints/aishell1_fc_rnnt_bpe_5000_50_n1_bs64_lr2.5e-4_a100_600M_spec00.nemo