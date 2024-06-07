#%%
import os
import sys
import re
import json
import jiwer
import soundfile as sf
import librosa
import numpy as np
from textgrid import TextGrid, IntervalTier
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict

import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

from einops import rearrange

# get audio files from Meta's Voicebox demo page
sampling_rate = 24000

from examples.tts.voicebox_demo import get_data, get_audio_data
from nemo.collections.asr.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_lhotse import LhotseTextToSpeechDataset
from nemo.collections.tts.models.voicebox import VoiceboxModel, fix_alignment
from nemo.collections.tts.modules.voicebox_modules import get_mask_from_lengths

from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm, trange

sys_prompt = """Given a transcript of a speech, your task is to subtly alter its meaning by changing only a single word or phrase. This change should pivot the original intent or information conveyed, such as transforming a word to its antonym, substituting a name or a noun with another, or slightly altering a phrase that shifts the narrative direction. The challenge lies in making this change impactful enough to alter the transcript's overall meaning, while ensuring that the modification is limited to a very small part of the text. The rest of the transcript must remain untouched.

**Objective:** Focus on identifying a pivotal word or phrase whose modification can flip the narrative or significantly change the message, with minimal intervention.

**Constraints:**
- Only one word or phrase may be altered.
- The alteration should substantially change the meaning of the transcript.
- All other words in the transcript must remain exactly as they are.
- The modified word or phrase should constitute a small ratio of the text to ensure the exercise's subtlety.

**Output Requirement:** Provide only the modified transcript. Do not include any explanations or annotations.

**Example:**

- **Original Transcript:** "There's a llama on my lawn, how can I get rid of him?"
- **Modified Transcript:** "There's a lion on my lawn, how can I get rid of him?"

Proceed by applying this instruction to the given transcript, ensuring the modification adheres to the outlined constraints and objectives."""

class DataProcessor:
    def __init__(self, model: VoiceboxModel):
        self.model = model

    def prepare_val_dl(self, ds_name="libriheavy", corpus_dir="/datasets/LibriLight/", manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_dev.jsonl.gz",
                       old_prefix="download/librilight", min_duration=-1, max_duration=float("inf"), load_audio=True, filter_ids=None):
        # load from val set
        self.model.cfg.ds_name = ds_name
        self.model.cfg.corpus_dir = corpus_dir
        self.model.cfg.validation_ds.manifest_filepath = manifest_filepath
        self.model.cfg.validation_ds.lhotse.cuts_path = self.model.cfg.validation_ds.manifest_filepath
        with open_dict(self.model.cfg.validation_ds):
            self.model.cfg.validation_ds.min_duration = min_duration
            self.model.cfg.validation_ds.max_duration = max_duration
            self.model.cfg.validation_ds.ds_kwargs.load_audio = load_audio
            self.model.cfg.validation_ds.filter_ids = filter_ids
            self.model.cfg.validation_ds.num_workers = 8
        with open_dict(self.model.cfg):
            self.model.cfg["old_prefix"] = old_prefix
        self.model.setup_validation_data(self.model.cfg.validation_ds)

    def get_val_batch(self, from_tb=False, mix=False, batch_idx=0):
        val_dl = self.model._validation_dl

        for i, batch in enumerate(val_dl):
            # print(batch.keys())
            if i == batch_idx:
                break

        audio = batch["audio"].to(self.model.device)
        audio_lens = batch["audio_lens"].to(self.model.device)
        audio_mask = get_mask_from_lengths(audio_lens)
        tokens = batch["tokens"].to(self.model.device)
        token_lens = batch["token_lens"].to(self.model.device)
        durations = batch.get("durations", None).to(self.model.device)
        if from_tb:
            audios = []
            audio_lens = []
            for i in range(4):
                _audio, _audio_len, _ = get_audio_data(f"nemo_experiments/tb_val/val_vb-{i}-orig_audio.wav", device=self.model.device)
                audios.append(_audio[0])
                audio_lens.append(_audio_len[0])
            audio = pad_sequence(audios, batch_first=True, padding_value=0)
            audio_lens = torch.stack(audio_lens)

        dp_inputs = self.model.duration_predictor.parse_dp_input(
            x1=audio,
            mask=audio_mask,
            durations=durations,
            phoneme_len=token_lens,
            input_sampling_rate=None,
        )

        tokens = self.model.duration_predictor.align_phoneme_ids_with_durations(tokens, dp_inputs.get("dp_cond"))

        vb_inputs = self.model.cfm_wrapper.parse_vb_input(
            x1=audio,
            mask=audio_mask,
            cond=audio,
            input_sampling_rate=None
        )
        x1 = vb_inputs['x1']
        cond = vb_inputs['cond']
        self_attn_mask = vb_inputs['mask']

        if not mix:
            self.model.voicebox.eval()
            cond_mask = self.model.voicebox.create_cond_mask(
                batch=cond.shape[0],
                seq_len=cond.shape[1],
                cond_token_ids=tokens,
                self_attn_mask=self_attn_mask,
                training=True,
                frac_lengths_mask=(0.1, 0.5),
            )
            # cond = cond * ~rearrange(cond_mask, '... -> ... 1')

            return {
                "texts": batch["texts"],
                "cuts": batch["cuts"],
                "alignments": [cut.supervisions[0].alignment for cut in batch["cuts"]],
                # "audio_paths": batch["audio_paths"],
                "ori_audio": audio,
                "ori_audio_lens": audio_lens,
                "ori_mel": x1,
                "cond": cond,
                "cond_mask": cond_mask.bool(),
                "tokens": tokens,
                "self_attn_mask": self_attn_mask.bool(),
            }
        
        # mix

    def find_word_mask(self, batch, dp_inputs):
        dp_cond = dp_inputs["dp_cond"]
        cum_dur = dp_inputs["cum_dur"]
        mel_len = dp_inputs["mel_len"]
        phn_ids = batch["tokens"].to(self.model.device)
        token_lens = batch["token_lens"].to(self.model.device)
        mid = torch.zeros_like(mel_len).float().uniform_(.25, .75) * mel_len
        mid_pos_idx = (mid.unsqueeze(1) >= cum_dur).sum(-1) + 1
        mid_phn_idx = phn_ids[range(len(mid)), mid_pos_idx]
        st = mid_pos_idx
        should_cont = (st > 0) & (mid_phn_idx > 4)
        # should_cont = (st > 0) & ~(mid_phn_idx < 5 & dp_cond[range(len(mid)), st] > 0)
        while should_cont.sum() > 0:
            prev_pos_idx = torch.where(should_cont, st - 1, st)
            prev_phn_idx = phn_ids[range(len(mid)), prev_pos_idx]
            st = torch.where(should_cont & (prev_phn_idx > 4), st - 1, st)
            should_cont = should_cont & (prev_phn_idx > 4) & (st > 0)
        ed = mid_pos_idx
        should_cont = (ed < token_lens-1)
        while should_cont.sum() > 0:
            post_pos_idx = torch.where(should_cont, ed + 1, ed)
            post_phn_idx = phn_ids[range(len(mid)), post_pos_idx]
            ed = torch.where(should_cont & (post_phn_idx > 4), ed + 1, ed)
            should_cont = should_cont & (post_phn_idx > 4) & (ed < token_lens-1)
        assert torch.all(ed > st)

    def get_demo_batch(self, corpus_dir, textgrid_dir, ori=False):
        outputs = get_data(self.model, corpus_dir, textgrid_dir, self.model.device)

        print(outputs['1']['ori_sr'])

        ori_audios = []
        ori_mels = []
        new_conds = []
        new_frame_masks = []
        new_aligned_phoneme_ids = []
        self_attn_masks = []
        for spk in tqdm(outputs):
            _ori_audio, ori_mel = outputs[spk]["ori_audio"], outputs[spk]["ori_mel"]
            _ori_audio, ori_mel = outputs[spk]["ori_audio"], outputs[spk]["ori_mel"]
            ori_self_attn_mask, ori_aligned_phoneme_ids = outputs[spk]["ori_self_attn_mask"], outputs[spk]["ori_aligned_phoneme_ids"]
            new_cond, new_frame_mask, aligned_phoneme_ids = outputs[spk]["new_cond"], outputs[spk]["new_frame_mask"], outputs[spk]["aligned_phoneme_ids"]

            # cut_st = new_phn_mask.index(0)
            # cut_ed = new_phn_mask.index(0, cut_st)
            # t_st_pos = int(new_phn_dur[cut_st][1] * sampling_rate)
            # t_ed_pos = int(new_phn_dur[cut_ed][2] * sampling_rate)
            # print(t_st_pos, t_ed_pos)
            # ori_spk_emb = voice_encoder.embed_utterance(ori_audio[t_st_pos:min(t_ed_pos, ori_audio.shape[-1])])

            # pad_cond = torch.ones_like(new_cond) * 0
            # pad_cond = torch.ones_like(new_cond) * -4.5252
            # pad_mask = torch.ones_like(new_frame_mask) * 1
            # new_cond = torch.cat([new_cond, pad_cond], dim=1)
            # new_frame_mask = torch.cat([new_frame_mask, pad_mask], dim=1)
            # aligned_phoneme_ids = torch.cat([aligned_phoneme_ids, aligned_phoneme_ids], dim=1)
            # aligned_phoneme_ids = torch.cat([aligned_phoneme_ids, torch.zeros_like(aligned_phoneme_ids)], dim=1)

            ori_audios.append(_ori_audio[0])
            ori_mels.append(ori_mel[0])
            if ori:
                new_conds.append(ori_mel[0])
                new_frame_masks.append(torch.zeros_like(ori_self_attn_mask[0]))
                new_aligned_phoneme_ids.append(ori_aligned_phoneme_ids[0])
                self_attn_masks.append(ori_self_attn_mask[0])
            else:
                new_conds.append(new_cond[0])
                new_frame_masks.append(new_frame_mask[0])
                new_aligned_phoneme_ids.append(aligned_phoneme_ids[0])
                self_attn_masks.append(torch.ones_like(new_frame_mask[0]))

        ori_audio = pad_sequence(ori_audios, batch_first=True, padding_value=0)
        ori_mel = pad_sequence(ori_mels, batch_first=True, padding_value=-4.5252)
        new_cond = pad_sequence(new_conds, batch_first=True, padding_value=-4.5252)
        new_frame_mask = pad_sequence(new_frame_masks, batch_first=True, padding_value=0)
        aligned_phoneme_ids = pad_sequence(new_aligned_phoneme_ids, batch_first=True, padding_value=0)
        self_attn_mask = pad_sequence(self_attn_masks, batch_first=True, padding_value=0).bool()

        if not ori:
            return {
                "ori_audio": ori_audio,
                "ori_mel": ori_mel,
                "cond": new_cond,
                "cond_mask": new_frame_mask.bool(),
                "tokens": aligned_phoneme_ids,
                "self_attn_mask": self_attn_mask.bool(),
            }

        self.model.voicebox.eval()
        cond = ori_mel
        cond_mask = self.model.voicebox.create_cond_mask(
            batch=new_cond.shape[0],
            seq_len=new_cond.shape[1],
            cond_token_ids=aligned_phoneme_ids,
            self_attn_mask=self_attn_mask,
            training=True,
            frac_lengths_mask=(0.1, 0.5),
        )

        return {
            "ori_audio": ori_audio,
            "ori_mel": ori_mel,
            "cond": cond,
            "cond_mask": cond_mask.bool(),
            "tokens": aligned_phoneme_ids,
            "self_attn_mask": self_attn_mask.bool(),
        }

    def fix_demo_audio(self, batch, scale=1.):
        cond_shift = torch.zeros_like(batch['cond'])
        # cond_shift[:, :1, -16:] = batch['cond'][:, :1, -32:-16].mean(-1, keepdim=True) - batch['cond'][:, :1, -16:].mean(-1, keepdim=True)
        cond_shift[:, :, -16:-12] = batch['cond'][:, 4:20, -32:-16].mean([-1, -2], keepdim=True) - batch['cond'][:, 4:20, -16:-12].mean([-2], keepdim=True)
        cond_shift[:, :, -12:] = batch['cond'][:, 4:20, -32:-16].mean([-1, -2], keepdim=True) - batch['cond'][:, 4:20, -12:].mean([-1, -2], keepdim=True)
        batch['cond'] = batch['cond'] + (cond_shift * scale)
        # batch['cond']: torch.Tensor = batch['cond'] * rearrange(batch['self_attn_mask'], '... -> ... 1')
        batch['cond'].masked_fill_(~rearrange(batch['self_attn_mask'], '... -> ... 1'), -4.5252)

        return batch

    def mask_cond(self, batch):
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        cond = cond * ~rearrange(cond_mask, '... -> ... 1')
        batch["cond"] = cond
        return batch

    def get_internal_demo_data(self, output_dir="nemo_experiments/internal_demo_gen"):
        os.makedirs(output_dir, exist_ok=True)
        datas = [
            {
                "audio_path": "nemo_experiments/internal_demo/Oh, I love that song/Oh, I love that song.wav",
                "text": "Oh, I love that song.",
                "textgrid_path": "nemo_experiments/internal_demo_mfa/Oh, I love that song/Oh, I love that song.TextGrid",
                "from": "love",
                "to": "hate",
                "out_ori_path": f"{output_dir}/Oh, I love that song_ori.wav",
                "out_gen_path": f"{output_dir}/Oh, I hate that song_gen.wav",
                "out_tts_path": f"{output_dir}/Oh, I hate that song_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/internal_demo/SEMamba_you know the lady gaga song at the end of Maverick_ori.wav",
                "text": "you know the lady gaga song at the end of Maverick",
                # "textgrid_path": "nemo_experiments/internal_demo_mfa/yoy know the lage gaga song at the end of Maverick/yoy know the lage gaga song at the end of Maverick.TextGrid",
                "from": "end",
                "to": "beginning",
                "out_ori_path": f"{output_dir}/SE_you know the lady gaga song at the end of Maverick_ori.wav",
                "out_gen_path": f"{output_dir}/SE_you know the lady gaga song at the beginning of Maverick_gen.wav",
                "out_tts_path": f"{output_dir}/SE_you know the lady gaga song at the beginning of Maverick_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/internal_demo/yoy know the lage gaga song at the end of Maverick/yoy know the lage gaga song at the end of Maverick.wav",
                "text": "you know the lady gaga song at the end of Maverick",
                "textgrid_path": "nemo_experiments/internal_demo_mfa/yoy know the lage gaga song at the end of Maverick/yoy know the lage gaga song at the end of Maverick.TextGrid",
                "from": "end",
                "to": "beginning",
                "out_ori_path": f"{output_dir}/you know the lady gaga song at the end of Maverick_ori.wav",
                "out_gen_path": f"{output_dir}/you know the lady gaga song at the beginning of Maverick_gen.wav",
                "out_tts_path": f"{output_dir}/you know the lady gaga song at the beginning of Maverick_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/internal_demo/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it.wav",
                # "audio_path": "nemo_experiments/internal_demo/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much money to go fill it_gen_SE.wav",
                "text": "by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it",
                "from": "information",
                "to": "money",
                "out_ori_path": f"{output_dir}/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it_ori.wav",
                "out_gen_path": f"{output_dir}/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much money to go fill it_gen.wav",
                "out_tts_path": f"{output_dir}/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much money to go fill it_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/internal_demo/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it.wav",
                # "audio_path": "nemo_experiments/internal_demo/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much money to go fill it_gen_SE.wav",
                "text": "by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it",
                "from": "months",
                "to": "milliseconds",
                "out_ori_path": f"{output_dir}/by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it_ori.wav",
                "out_gen_path": f"{output_dir}/by doubling every six milliseconds if you double the size of the model you double the size of your brain you need twice as much information to go fill it_gen.wav",
                "out_tts_path": f"{output_dir}/by doubling every six milliseconds if you double the size of the model you double the size of your brain you need twice as much information to go fill it_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/REAL/dev_real_medium-103-eldorado_krs_librivox_64kb_mp3-eldorado_37_orczy_64kb_11.wav",
                "text": "here the noise and hubbub that went on constantly in the guardroom would effectually drown a whispered conversation",
                "from": "noise",
                "to": "silence",
                "out_ori_path": f"{output_dir}/dev_real_medium-103-eldorado_krs_librivox_64kb_mp3-eldorado_37_orczy_64kb_11_ori.wav",
                "out_gen_path": f"{output_dir}/dev_real_medium-103-eldorado_krs_librivox_64kb_mp3-eldorado_37_orczy_64kb_11_gen.wav",
                "out_tts_path": f"{output_dir}/dev_real_medium-103-eldorado_krs_librivox_64kb_mp3-eldorado_37_orczy_64kb_11_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/REAL/dev_real_medium-107-boys_life_of_twain_jg_0810_librivox_64kb_mp3-twain_61_paine_64kb_16.wav",
                "text": "following his birthday dinner mark twain had become once more the Belle of new york and in a larger way than ever before",
                "from": "more",
                "to": "again",
                "out_ori_path": f"{output_dir}/dev_real_medium-107-boys_life_of_twain_jg_0810_librivox_64kb_mp3-twain_61_paine_64kb_16_ori.wav",
                "out_gen_path": f"{output_dir}/dev_real_medium-107-boys_life_of_twain_jg_0810_librivox_64kb_mp3-twain_61_paine_64kb_16_gen.wav",
                "out_tts_path": f"{output_dir}/dev_real_medium-107-boys_life_of_twain_jg_0810_librivox_64kb_mp3-twain_61_paine_64kb_16_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/REAL/dev_real_medium-803-stories_006_librivox_64kb_mp3-damned_thing_bierce_ge_64kb_31.wav",
                "text": "the men removed their hats the witness was sworn what is your name the coroner asked",
                "from": "name",
                "to": "nickname",
                "out_ori_path": f"{output_dir}/dev_real_medium-803-stories_006_librivox_64kb_mp3-damned_thing_bierce_ge_64kb_31_ori.wav",
                "out_gen_path": f"{output_dir}/dev_real_medium-803-stories_006_librivox_64kb_mp3-damned_thing_bierce_ge_64kb_31_gen.wav",
                "out_tts_path": f"{output_dir}/dev_real_medium-803-stories_006_librivox_64kb_mp3-damned_thing_bierce_ge_64kb_31_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/REAL/dev_real_medium-2691-expedition_donner_party_0907_librivox_64kb_mp3-expeditiondonner_17_houghton_64kb_26.wav",
                "text": "on our way out a neighbor intercepted us and said that we should sleep at her house that night and see our sisters in the morning",
                "from": "see",
                "to": "seeing",
                "out_ori_path": f"{output_dir}/dev_real_medium-2691-expedition_donner_party_0907_librivox_64kb_mp3-expeditiondonner_17_houghton_64kb_26_ori.wav",
                "out_gen_path": f"{output_dir}/dev_real_medium-2691-expedition_donner_party_0907_librivox_64kb_mp3-expeditiondonner_17_houghton_64kb_26_gen.wav",
                "out_tts_path": f"{output_dir}/dev_real_medium-2691-expedition_donner_party_0907_librivox_64kb_mp3-expeditiondonner_17_houghton_64kb_26_tts.wav",
            },
            {
                "audio_path": "nemo_experiments/REAL/dev_real_medium-26-byways_around_san_francisco_bay_librivox_64kb_mp3-bywaysaroundsf_19_hutchinson_64kb_8.wav",
                "text": "It has its moods both grave and gay and is as fickle as a schoolgirl",
                "from": "schoolgirl",
                "to": "teenager",
                "out_ori_path": f"{output_dir}/dev_real_medium-26-byways_around_san_francisco_bay_librivox_64kb_mp3-bywaysaroundsf_19_hutchinson_64kb_8_ori.wav",
                "out_gen_path": f"{output_dir}/dev_real_medium-26-byways_around_san_francisco_bay_librivox_64kb_mp3-bywaysaroundsf_19_hutchinson_64kb_8_gen.wav",
                "out_tts_path": f"{output_dir}/dev_real_medium-26-byways_around_san_francisco_bay_librivox_64kb_mp3-bywaysaroundsf_19_hutchinson_64kb_8_tts.wav",
            },
        ]
        return datas

    def get_word_edit_data(self):
        datas = {
            "libriheavy": [
                {
                    "sample_id": 4,
                    "edit_from": "last",
                    "edit_to": "first",
                },
                {
                    "sample_id": 6,
                    "edit_from": "private",
                    "edit_to": "public",
                },
            ],
            "gigaspeech": [
                {
                    "sample_id": 0,
                    "edit_from": "biggest",
                    "edit_to": "smallest",
                },
                {
                    "sample_id": 1,
                    "edit_from": "include",
                    "edit_to": "exclude",
                },
                {
                    "sample_id": 2,
                    "edit_from": "address",
                    "edit_to": "password",
                },
                {
                    "sample_id": 5,
                    "edit_from": "percentage",
                    "edit_to": "amount",
                },
            ],
        }
        return datas

class Inference:
    def __init__(self, model: VoiceboxModel):
        self.model = model

    def test_batch(self, batch, out_dir, prefix=""):
        os.makedirs(out_dir, exist_ok=True)

        self_attn_mask = batch["self_attn_mask"]
        tokens = batch["tokens"]
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        ori_audios = batch["ori_audio"]

        cond = cond * ~rearrange(cond_mask, '... -> ... 1')

        self.model.voicebox.eval()
        masked_audios = self.model.voicebox.audio_enc_dec.decode(cond)
        output_mels = self.model.cfm_wrapper.sample(
            cond=cond,
            self_attn_mask=self_attn_mask,
            aligned_phoneme_ids=tokens,
            cond_mask=cond_mask,
            steps=100,
            decode_to_audio=False
        )
        output_audios = self.model.voicebox.audio_enc_dec.decode(output_mels)

        for i in trange(ori_audios.shape[0]):
            # ori_audio = self.model.voicebox.audio_enc_dec.decode(ori_mel)
            ori_audio = ori_audios[i].cpu().numpy()
            sf.write(f"{out_dir}/{prefix}{i}_0.wav", ori_audio, sampling_rate, format='WAV')

            output_audio = output_audios[i].cpu().numpy()
            masked_audio = masked_audios[i].cpu().numpy()

            # output_audio = output_audio / max(np.abs(output_audio))
            sf.write(f"{out_dir}/{prefix}{i}_1_pred.wav", output_audio, sampling_rate, format='WAV')

            # masked_audio = masked_audio / max(np.abs(masked_audio))
            sf.write(f"{out_dir}/{prefix}{i}_0_masked.wav", masked_audio, sampling_rate, format='WAV')
        return output_mels

    def test_demo_batch(self, dataprocessor: DataProcessor, corpus_dir, textgrid_dir, out_dir):
        # max_sims = {'0': 0.9, '1': 0.89, '2': 0.86, '3': 0.89, '4': 0.858, '5': 0.88,}

        demo_batch = dataprocessor.get_demo_batch(corpus_dir, textgrid_dir)
        self.test_batch(demo_batch, out_dir)

        # for spk in tqdm(spks):
        #     # print(output_audio.shape)
        #     # max_sim = max_sims[spk]
        #     output_audio = output_audios[int(spk)].cpu().numpy()
        #     masked_audio = masked_audios[int(spk)].cpu().numpy()

        #     # output_audio = output_audio / max(np.abs(output_audio))
        #     # sim = voice_encoder.embed_utterance(output_audio[t_st_pos:min(t_ed_pos, output_audio.shape[-1])]) @ ori_spk_emb
        #     # tqdm.write(f"{sim}")
        #     # if sim > max_sim:
        #         # max_sim = sim
        #         # sf.write(f"{out_dir}/{spk}_{sim:.7f}.wav", output_audio, sampling_rate, format='WAV')
        #         # print(f"{out_dir}/{spk}_{sim:.7f}.wav")

        #         # masked_audio = masked_audio / max(np.abs(masked_audio))
        #         # sf.write(f"{out_dir}/{spk}_{sim:.7f}_0_masked.wav", masked_audio, sampling_rate, format='WAV')

        #     # masked_audio = masked_audio / max(np.abs(masked_audio))
        #     sf.write(f"{out_dir}/{spk}_1_pred.wav", output_audio, sampling_rate, format='WAV')
        #     sf.write(f"{out_dir}/{spk}_0_masked.wav", masked_audio, sampling_rate, format='WAV')

    def word_edit(self, dataprocessor: DataProcessor, output_dir="nemo_experiments/edit_gen/"):
        val_batch = dataprocessor.get_val_batch(batch_idx=0)
        datas = dataprocessor.get_word_edit_data()[self.model.cfg.ds_name]
        os.makedirs(output_dir, exist_ok=True)

        for data in datas:
            sample_id = data["sample_id"]
            audio_len = val_batch["ori_audio_lens"][sample_id].unsqueeze(0)
            audio = val_batch["ori_audio"][sample_id, :audio_len].unsqueeze(0)
            audio_id = val_batch["cuts"][sample_id].id
            edit_pred = self.model.forward(
                audio=audio,
                audio_lens=audio_len,
                texts=[val_batch["texts"][sample_id]],
                alignments=[val_batch["alignments"][sample_id]],
                edit_from=[data["edit_from"]],
                edit_to=[data["edit_to"]],
                steps=16,
            )
            edit_audio = edit_pred["edit_audio"]
            ztts_audio = edit_pred["ztts_audio"]
            sf.write(f"{output_dir}/val_{audio_id}_ori.wav", audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
            sf.write(f"{output_dir}/val_{audio_id}_gen.wav", edit_audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
            sf.write(f"{output_dir}/val_{audio_id}_tts.wav", ztts_audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

    def internal_demo(self, data):
        # shape: (1, L), (1,), scalar
        # audio, audio_len, orig_sr = get_audio_data(data["audio_path"], self.model.device)
        # _, orig_sr = librosa.load(data["audio_path"], sr=None)
        audio_data, _sr = librosa.load(data["audio_path"], sr=self.model.voicebox.audio_enc_dec.sampling_rate)
        audio = torch.tensor(audio_data, dtype=torch.float, device=self.model.device).unsqueeze(0)
        audio_len = torch.tensor(audio.shape[1], device=self.model.device).unsqueeze(0)
        # audio = audio / 2
        
        edit_pred = self.model.forward(
            audio=audio,
            audio_lens=audio_len,
            texts=[data["text"],],
            textgrids=None if "textgrid_path" not in data else [data["textgrid_path"],],
            edit_from=[data["from"],],
            edit_to=[data["to"],],
            steps=64,
            cond_scale=1.0,
            sample_std=1.0,
            dp_scale=1.2,
        )
        edit_audio = edit_pred["edit_audio"][0].cpu().numpy()
        ztts_audio = edit_pred["ztts_audio"][0].cpu().numpy()
        # edit_audio = edit_audio / max(np.abs(edit_audio))
        sf.write(data["out_ori_path"], audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
        sf.write(data["out_gen_path"], edit_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
        sf.write(data["out_tts_path"], ztts_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
        return edit_pred["ori_mel"], edit_pred["edit_mel"]

class DataGen:
    def __init__(self, model: VoiceboxModel):
        self.model = model

    def gen_v1_dataset_from_val_set(self, out_dir):
        val_dl = self.model._validation_dl

        os.makedirs(f"{out_dir}/combine", exist_ok=True)
        label = [[], []] #fake, real

        for batch in tqdm(val_dl):
            batch = self.model.transfer_batch_to_device(batch, self.model.device, 0)
            batch = self.model.parse_val_vb_input(batch)

            cuts = batch["cuts"]
            ori_audio = batch["audio"]
            ori_mel = batch["mel"]
            ori_mel_lens = batch['mel_lens']
            cond = batch["cond"]    # same as ori_mel
            cond_mask = batch["cond_mask"]
            aligned_tokens = batch["aligned_tokens"]
            self_attn_mask = batch["self_attn_mask"]

            cond_st_idx = torch.arange(cond.shape[1], 0, -1, device=self.model.device).reshape(1, -1) * cond_mask
            cond_st = cond_st_idx.argmax(dim=1)
            cond_ed_idx = torch.arange(cond.shape[1], device=self.model.device).reshape(1, -1) * cond_mask
            cond_ed = cond_ed_idx.argmax(dim=1) + 1

            assert cond.shape[0] == cond_mask.shape[0] and cond.shape[1] == cond_mask.shape[1], \
                f"{cond.shape}, {cond_mask.shape}, {self_attn_mask.shape}, {aligned_tokens.shape}"
            try:
                gen_audio = self.model.cfm_wrapper.sample(
                    cond=cond,
                    self_attn_mask=self_attn_mask.bool(),
                    aligned_phoneme_ids=aligned_tokens,
                    cond_mask=cond_mask.bool(),
                    steps=16,
                )
            except:
                print("X")
                continue
            ori_audio_lens = batch["audio_lens"]
            gen_audio_lens = torch.clamp(ori_audio_lens, max=gen_audio.shape[-1])
            ori_ls = ori_audio_lens / 24000
            gen_ls = gen_audio_lens / 24000
            gen_st = gen_ls * cond_st / ori_mel_lens
            gen_ed = gen_ls * cond_ed / ori_mel_lens

            for i in range(cond.shape[0]):
                new_id = '-'.join(cuts[i].id.split('/'))
                if cond_st[i] == 0:
                    label[0].append(f"dev_fake_{new_id} {gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                elif cond_ed[i] == ori_mel_lens[i]:
                    label[0].append(f"dev_fake_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F 0")
                else:
                    label[0].append(f"dev_fake_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                label[1].append(f"dev_real_{new_id} 0.00-{ori_ls[i]:.2f}-T 1")
                # print(label[0][-1])
                # print(label[1][-1])

                _ori_audio = ori_audio[i, :ori_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_real_{new_id}.wav", _ori_audio, sampling_rate, format='WAV')
                _gen_audio = gen_audio[i, :gen_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_fake_{new_id}.wav", _gen_audio, sampling_rate, format='WAV')

        with open(f"{out_dir}/dev_label.txt", 'w') as f:
            f.write('\n'.join(label[0]))
            f.write('\n')
            f.write('\n'.join(label[1]))

    @staticmethod
    def normalize_text(text):
        # Define the punctuation to strip, excluding brackets that should be preserved
        # punctuation_to_strip = r"[、।，@”,:;¿?¡!\\&%#*~，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=]+"
        punctuation_to_strip = r"[、。।，@<>”(),.:;¿?¡!\&%#*~【】，…‥「」『』\"\'_〝〟″⟨⟩♪・‹›«»～′$+=]+"
        
        # Define brackets that should be preserved if they enclose the whole word
        brackets_to_preserve = r"(\[\w+\])|(\{\w+\})|(<\w+>)|(\(\w+\))|(＜\w+＞)"
        
        # Split the text into words using whitespace and additional word break markers, preserving words within brackets
        word_break_markers = r"[\s？!()，,.:;¡¿?“„”&~%#—…‥、。【】$+=〝〟″‹›«»・⟨⟩「」『』”]+"
        words = re.split(word_break_markers, text)
        
        normalized_words = []
        for word in words:
            # Check if the word is enclosed in brackets that should be preserved
            if re.match(brackets_to_preserve, word):
                normalized_words.append(word.lower())
            else:
                # Strip specified punctuation from the beginning and end, then lowercase the word
                word = re.sub(f"^{punctuation_to_strip}|{punctuation_to_strip}$", "", word)
                normalized_words.append(word.lower())
        
        # Rejoin the normalized words into a single string
        return ' '.join(normalized_words)

        # Example usage, commented out to prevent execution
        # text = "This is an example text, with various punctuation marks! Including: brackets (like these)."
        # normalized_text = normalize_text(text)
        # print(normalized_text)

    @staticmethod
    def filter_alignments(alignments: list[jiwer.AlignmentChunk]):
        # if len(alignments) == 1:
        #     return alignments[0].type == "equal"
        # elif len(alignments) == 2:
        #     for ali in alignments:
        #         if ali.type not in ["equal", "substitute"]:
        #             return False
        #     return True
        # elif len(alignments) == 3:
        #     return [ali.type for ali in alignments] == ["equal", "substitute", "equal"]
        # else:
        for ali in alignments:
            if ali.type == "substitute":
                return True
        return False

    def gen_edit_transcript_json(self, out_json_path):
        val_dl = self.model._validation_dl

        out_file = open(out_json_path, 'w')

        for batch in tqdm(val_dl):
            texts = batch["texts"]
            cuts = batch["cuts"]
            for cut, text in zip(cuts, texts):
                alignment = cut.supervisions[0].alignment
                alignment = fix_alignment(alignment)
                words = [ali.symbol for ali in alignment["words"] if ali.symbol != "<eps>"]
                phns = [ali.symbol for ali in alignment["phones"] if ali.symbol != "sil"]
                # text = ' '.join(words)
                if "<unk>" in words:
                    # print(words)
                    continue
                if "spn" in phns:
                    # print(phns)
                    continue
                out = {
                    "id": cut.id,
                    "sys_prompt": sys_prompt,
                    "prompt": text,
                }
                json.dump(out, out_file)
                out_file.write('\n')

        out_file.close()

    def load_gpt_json(self, json_filename, out_filename):
        mfa_en_dict = {}
        with open("/root/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict", 'r') as f:
            for line in tqdm(f):
                wrd, _, _, _, _, phns = line.strip().split('\t')
                if wrd not in mfa_en_dict:
                    mfa_en_dict[wrd] = phns

        from collections import Counter
        import random
        out_dict = {}
        with open(json_filename, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            line = json.loads(line)
            assert line['id'] not in out_dict, line
            # ref = line['prompt']
            # hyp = line['gpt_text']
            # ref = self.normalize(line['prompt'])
            # hyp = self.normalize(line['gpt_text'])
            ref = self.normalize_text(line['prompt'])
            hyp = self.normalize_text(line['gpt_text'])
            out = jiwer.process_words(ref, hyp)
            _ref = out.references[0]
            _hyp = out.hypotheses[0]

            if not self.filter_alignments(out.alignments[0]):
                continue

            alis = [ali for ali in out.alignments[0] if ali.type == "substitute"]
            ali = random.choice(alis)
            froms = [_ref[i] for i in range(ali.ref_start_idx, ali.ref_end_idx)]
            tos = [_hyp[i] for i in range(ali.hyp_start_idx, ali.hyp_end_idx)]
            to = random.choice(tos)
            try:
                if to in mfa_en_dict:
                    phns = mfa_en_dict[to].split(' ')
                else:
                    phns = os.popen(f"conda run -n aligner bash -c \"echo '{to}' | mfa g2p -n 1 - english_us_arpa - 2> /dev/null\"").read().split('\t')[1].strip().split(' ')
            except:
                continue
            out_dict[line['id']] = {
                "ref": ' '.join(_ref),
                "hyp": ' '.join(_hyp),
                "froms": ' '.join(froms),
                "tos": ' '.join(tos),
                "to": to,
                "to_phns": phns,
            }
        with open(out_filename, 'w') as fo:
            json.dump(out_dict, fo, indent=4)
        return out_dict

    def gen_v3_dataset_from_val_set(self, edit_dict, out_dir):
        import random
        val_dl = self.model._validation_dl

        os.makedirs(f"{out_dir}/combine", exist_ok=True)
        label = {"edit": [], "cut_paste": [], "resyn": [], "real": []} #fake, real

        f_real = open(f"{out_dir}/medium_real.txt", 'w')
        f_resyn = open(f"{out_dir}/medium_resyn.txt", 'w')
        f_edit = open(f"{out_dir}/medium_edit.txt", 'w')
        f_cut_paste = open(f"{out_dir}/medium_cut_paste.txt", 'w')

        count = 0
        for batch in tqdm(val_dl):
            batch = self.model.transfer_batch_to_device(batch, self.model.device, 0)
            # batch = self.model.parse_val_vb_input(batch)

            # alignments: from cuts
            # audio, audio_lens
            # mel, mel_lens

            cuts = batch["cuts"]
            alignments = []
            edit_froms = []
            edit_tos = []
            indexes = []
            for i, c in tqdm(enumerate(cuts), leave=False):
                assert c.id in edit_dict
                alignment = c.supervisions[0].alignment
                froms = edit_dict[c.id]["froms"]
                if "to" in edit_dict[c.id] and "to_phns" in edit_dict[c.id]:
                    to = (edit_dict[c.id]["to"], edit_dict[c.id]["to_phns"])
                else:
                    tos = edit_dict[c.id]["tos"]
                    to = random.choice(tos.split(' '))
                mfa_wrds = [ali.symbol for ali in alignment["words"]]
                froms = [wrd for wrd in froms.split(' ') if wrd in mfa_wrds]
                if len(froms) > 0:
                    indexes.append(i)
                    alignments.append(alignment)
                    edit_froms.append(random.choice(froms))
                    edit_tos.append(to)
            indexes = torch.tensor(indexes, device=self.model.device)

            ori_audio = torch.index_select(batch["audio"], 0, indexes)
            ori_audio_lens = torch.index_select(batch["audio_lens"], 0, indexes)
            ori_texts = [t for i, t in enumerate(batch["texts"]) if i in indexes]

            try:
                pred = self.model.forward(
                    audio=ori_audio,
                    audio_lens=ori_audio_lens,
                    texts=ori_texts,
                    alignments=alignments,
                    edit_from=edit_froms,
                    edit_to=edit_tos,
                    steps=16,
                    # decode_to_audio=False,
                )
            except:
                print("X")
                continue
            edit_audio = pred["edit_audio"]
            cap_audio = pred["cap_audio"]
            resyn_audio = pred["resyn_audio"]
            gen_audio_lens = pred["new_audio_lens"]
            ori_ls = ori_audio_lens / 24000
            gen_ls = gen_audio_lens / 24000
            gen_st_idx = pred["new_cond_st_idx"]
            gen_ed_idx = pred["new_cond_ed_idx"]
            gen_st = gen_st_idx / 24000
            gen_ed = gen_ed_idx / 24000
            for i in range(edit_audio.shape[0]):
                _i = indexes[i].item()
                new_id = '-'.join(cuts[_i].id.split('/'))

                if gen_st_idx[i] == 0:
                    label["edit"].append(f"dev_edit_{new_id} {gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                    label["cut_paste"].append(f"dev_cut_paste_{new_id} {gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                elif gen_ed_idx[i] == gen_audio_lens[i]:
                    label["edit"].append(f"dev_edit_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F 0")
                    label["cut_paste"].append(f"dev_cut_paste_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F 0")
                else:
                    label["edit"].append(f"dev_edit_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                    label["cut_paste"].append(f"dev_cut_paste_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                label["real"].append(f"dev_real_{new_id} 0.00-{ori_ls[i]:.2f}-T 1")
                label["resyn"].append(f"dev_resyn_{new_id} 0.00-{ori_ls[i]:.2f}-T 1")
                # print(label[0][-1])
                # print(label[1][-1])

                _ori_audio = ori_audio[i, :ori_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_real_{new_id}.wav", _ori_audio, sampling_rate, format='WAV')
                _resyn_audio = resyn_audio[i, :ori_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_resyn_{new_id}.wav", _resyn_audio, sampling_rate, format='WAV')
                _edit_audio = edit_audio[i, :gen_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_edit_{new_id}.wav", _edit_audio, sampling_rate, format='WAV')
                _cut_paste_audio = cap_audio[i, :gen_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_cut_paste_{new_id}.wav", _cut_paste_audio, sampling_rate, format='WAV')

                f_real.write(label["real"][-1] + '\n')
                f_resyn.write(label["resyn"][-1] + '\n')
                f_edit.write(label["edit"][-1] + '\n')
                f_cut_paste.write(label["cut_paste"][-1] + '\n')

                f_real.flush()
                f_resyn.flush()
                f_edit.flush()
                f_cut_paste.flush()

        # with open(f"{out_dir}/dev_label.txt", 'w') as f:
        #     f.write('\n'.join(label[0]))
        #     f.write('\n')
        #     f.write('\n'.join(label[1]))
        f_real.close()
        f_resyn.close()
        f_edit.close()
        f_cut_paste.close()
        return label

    @staticmethod    
    def normalize(sentence):
        sentence = sentence.lower()
        
        # remove all punctuation except words and space
        sentence = re.sub(r'[^\w\s]','', sentence)

        sentence = sentence.strip()
        return sentence

class Eval:
    def __init__(self,):
        self.voice_encoder = VoiceEncoder()

    @staticmethod
    def create_plot(data, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(data, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(data):
        Eval.create_plot(data.T.cpu(), 'frame', 'freq')

    @staticmethod
    def preprocess_wav(fpath_or_wav: str | np.ndarray, source_sr: int | None = None):
        """modified from resemblyzer.process_wav, remove trimming process"""
        from resemblyzer.audio import normalize_volume
        # Load the wav from disk if needed
        if isinstance(fpath_or_wav, str):
            wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
        else:
            wav = fpath_or_wav
        
        # Resample the wav
        if source_sr is not None:
            wav = librosa.resample(wav, orig_sr=source_sr, target_sr=16000)
            
        # Apply the preprocessing: normalize volume and shorten long silences 
        wav = normalize_volume(wav, -30, increase_only=True)
        # wav = trim_long_silences(wav)
        
        return wav

    def gen_val_frame_spk_sim(self, data_dir, subset="medium", audio_type="edit", file_type=None):
        """Gen {data_dir}/{subset}_{audio_type}_sim.txt for each {data_dir}/{subset}_{file_type}.txt"""
        file_type = audio_type if file_type is None else file_type
        with open(f"{data_dir}/{subset}_{file_type}.txt", 'r') as f, open(f"{data_dir}/{subset}_{audio_type}_sim.txt", 'w') as fo:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                fakename, info, label = line.split(' ')
                if line.startswith(f"dev_{audio_type}"):
                    _fname, fname_ = fakename.split('-', 1)
                    filename = f"{_fname.split('_')[-1]}-{fname_}"
                    realname = f"dev_real_{filename}"
                    fake_audio_path = f"{data_dir}/combine/{fakename}.wav"
                    real_audio_path = f"{data_dir}/combine/{realname}.wav"
                    metadata = {
                        "fake_audio_path": fake_audio_path,
                        "real_audio_path": real_audio_path,
                        "info": info,
                    }
                    fake_wav = self.preprocess_wav(fake_audio_path)
                    real_wav = preprocess_wav(real_audio_path)
                    spans = []
                    for span in info.split('/'):
                        span = span.split('-')
                        span, _label = [float(t)*16000 for t in span[:2]], span[2]
                        if _label == "F":
                            spans = span
                            break
                    speaker_emb = self.voice_encoder.embed_utterance(real_wav)
                    _, fake_frame_embs, wav_splits = self.voice_encoder.embed_utterance(fake_wav, return_partials=True, rate=25)
                    frame_sims = fake_frame_embs @ speaker_emb
                    sims = {"bound": [], "in": [], "out": [], "around": []}
                    for i, wav_split in enumerate(wav_splits):
                        if wav_split.start < spans[0]:
                            if wav_split.stop < spans[0]:
                                sims["out"].append(frame_sims[i])
                            else:
                                sims["bound"].append(frame_sims[i])
                                sims["around"].append(frame_sims[i])
                        elif wav_split.start < spans[1]:
                            if wav_split.stop < spans[1]:
                                sims["in"].append(frame_sims[i])
                                sims["around"].append(frame_sims[i])
                            else:
                                sims["bound"].append(frame_sims[i])
                                sims["around"].append(frame_sims[i])
                        else:
                            sims["out"].append(frame_sims[i])
                    for key in sims:
                        sims[key] = np.mean(sims[key]).item()
                                
                    # metadata["sims"] = sims
                    # metadata["sims"] = [sims["around"], sims["out"]]
                    # json.dump(metadata, fo)
                    fo.write(' '.join([fakename, info, f"{sims['around']:.3f}", f"{sims['out']:.3f}"]))
                    fo.write('\n')
                    fo.flush()

    def calc_sims(self):
        sims_dir = "/ssd2/sungfengh/gen_dataset/medium-v3/sims"
        in_sims = []
        out_sims = []
        diff_sims = []
        for i in trange(32):
            with open(f"{sims_dir}/split-{i}/medium_edit_sim.txt", 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    fakename, info, in_sim, out_sim = line.split(' ')
                    in_sim = float(in_sim)
                    out_sim = float(out_sim)
                    diff_sim = out_sim - in_sim
                    in_sims.append(in_sim)
                    out_sims.append(out_sim)
                    diff_sims.append(diff_sim)
        bin_edges = np.arange(.5, 1.01, .05)
        for _b, b_ in zip(bin_edges[:-1], bin_edges[1:]):
            print(_b, b_)
        in_hist = np.histogram(in_sims, bin_edges)
        out_hist = np.histogram(out_sims, bin_edges)
        _bin_edges = np.arange(.5, 1.01, .05)
        diff_hist = np.histogram(diff_sims, _bin_edges)

#%%
# if __name__ == "__main__":
    # ckpt_path = "nemo_experiments/local-1/2023-12-14_03-54-42/checkpoints/local-1--val_loss_total=9.0312-epoch=20-last.ckpt"
    # ckpt_path = "nemo_experiments/ngc/2023-12-15_16-41-45/checkpoints/ngc--val_loss_total=3.7990-epoch=20.ckpt"
    # ckpt_path = "nemo_experiments/ngc/2023-12-15_16-41-45/checkpoints/ngc--val_loss_total=3.7464-epoch=26.ckpt"
    # ckpt_path = "nemo_experiments/local_1-loss_full/2024-01-04_16-57-40/checkpoints/local_1-loss_full--val_loss_total=3.6258-epoch=7.ckpt"
    # ckpt_path = "nemo_experiments/local_1-loss_full/2024-01-05_14-50-46/checkpoints/local_1-loss_full--val_loss_total=3.5205-epoch=9.ckpt"

class MainExc:
    def __init__(self):
        # Mel + LibriLight
        # self.vb_ckpt_path = "nemo_experiments/vb=0.2669-epoch=15-last.ckpt"
        self.vb_ckpt_path = "nemo_experiments/vb=0.2573-epoch=42-last.ckpt"
        self.dp_ckpt_path = "nemo_experiments/1b_oci_voicebox--val_loss_total=3.2725-epoch=61.ckpt"

        # DAC + GS
        # self.vb_ckpt_path = "nemo_experiments/vb=0.7689-epoch=0-step=75932-last-001.ckpt"
        self.vb_ckpt_path = "nemo_experiments/vb=0.7526-epoch=0-step=130000.ckpt"
        # self.vb_ckpt_path = "nemo_experiments/vb=0.7406-epoch=0-step=163461-last.ckpt"
        self.dp_ckpt_path = "nemo_experiments/dp_no_sil_spn=1.4410-epoch=8.ckpt"

        self.gen_data_dir = "data/gen_dataset"

    def load_model(self,):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VoiceboxModel.load_from_checkpoint(self.vb_ckpt_path, map_location=device)

        # dp_state_dict = torch.load(self.dp_ckpt_path, map_location=device)["state_dict"]
        # model.load_part_of_state_dict(state_dict=dp_state_dict, include=["duration_predictor"], exclude=[], load_from_string=dp_ckpt_path)

        dp_model = VoiceboxModel.load_from_checkpoint(self.dp_ckpt_path, map_location=device)

        del model.duration_predictor, model.cfm_wrapper.duration_predictor
        model.duration_predictor = dp_model.duration_predictor
        model.cfm_wrapper.duration_predictor = dp_model.duration_predictor
        del dp_model

        # model = model.to(device)
        torch.cuda.empty_cache()
        return model

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.load_model()
        return self._model

    @property
    def dataprocessor(self):
        if not hasattr(self, "_dataprocessor"):
            self._dataprocessor = DataProcessor(model=self.model)
        return self._dataprocessor

    @property
    def datagen(self):
        if not hasattr(self, "_datagen"):
            self._datagen = DataGen(model=self.model)
        return self._datagen

    @property
    def infer(self):
        if not hasattr(self, "_infer"):
            self._infer = Inference(model=self.model)
        return self._infer
    
    @property
    def eval(self):
        if not hasattr(self, "_eval"):
            self._eval = Eval()
        return self._eval

    def gen_val_v1(self,):
        self.dataprocessor.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_dev.jsonl.gz")
        self.datagen.gen_v1_dataset_from_val_set("/datasets/LibriLight_aligned/gen_dataset/dev-v1")
        self.eval.gen_val_frame_spk_sim(data_dir="/datasets/LibriLight_aligned/gen_dataset/dev", subset="dev", audio_type="fake", file_type="label")

    def gen_v3_transcript_json(self):
        """generate json for LLM transcript editing"""
        self.dataprocessor.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_small.jsonl.gz", min_duration=4, max_duration=10, load_audio=False)
        self.datagen.gen_edit_transcript_json("/datasets/LibriLight_aligned/gen_dataset/small_prompt.json")

        self.dataprocessor.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_medium.jsonl.gz", min_duration=6, max_duration=8, load_audio=False)
        self.datagen.gen_edit_transcript_json("/datasets/LibriLight_aligned/gen_dataset/medium_prompt.json")

    def gen_v3(self, split_id=None, out_dict=None,
                   gpt_file="nemo_experiments/data_1a_medium.json", out_dict_file="nemo_experiments/data_parsed_1a_medium.json"):
        if not split_id:
            split_id = int(sys.argv[1])
        if not out_dict:
            if not os.path.exists(out_dict_file):
                out_dict = self.datagen.load_gpt_json(gpt_file, out_dict_file)
            else:
                out_dict = json.load(open(out_dict_file, 'r'))

        filter_ids = sorted(out_dict.keys())[split_id*3000: (split_id+1)*3000]
        self.dataprocessor.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_medium.jsonl.gz",
                                     min_duration=6, max_duration=8,
                                     filter_ids=filter_ids)
        self.datagen.gen_v3_dataset_from_val_set(out_dict, f"{self.gen_data_dir}/medium-v3/split-{split_id}")

        # split_id = int(sys.argv[1])
        self.eval.gen_val_frame_spk_sim(data_dir=f"{self.gen_data_dir}/medium-v3/split-{split_id}", subset="medium", audio_type="edit")
        self.eval.gen_val_frame_spk_sim(data_dir=f"{self.gen_data_dir}/medium-v3/split-{split_id}", subset="medium", audio_type="cut_paste")
        # exit()

    def v4_gs_val_word_edit(self, ds_name="gigaspeech", corpus_dir="data/download/GigaSpeech", manifest_filepath="data/parsed/GigaSpeech/gigaspeech_cuts_DEV.speech.jsonl.gz", output_dir="nemo_experiments/edit_gen/"):
        self.dataprocessor.prepare_val_dl(ds_name=ds_name, corpus_dir=corpus_dir, manifest_filepath=manifest_filepath, old_prefix="/home/sungfengh/.cache/huggingface/datasets")
        self.infer.word_edit(dataprocessor=self.dataprocessor, output_dir=output_dir)

    def _internal_demo(self, output_dir="nemo_experiments/internal_demo_gen"):
        datas = self.dataprocessor.get_internal_demo_data(output_dir)
        for data in datas:
            ori_mel, edit_mel = self.infer.internal_demo(data)
            self.eval.plot_spectrogram(ori_mel)
            self.eval.plot_spectrogram(edit_mel)

task = "edit"
task = "gendata"

corpus_dir = "/datasets/LibriLight_aligned/raw_data_cuts/demo"
textgrid_dir = "/datasets/LibriLight_aligned/textgrids/demo"
out_dir = "nemo_experiments/edit_demo"

main_exc = MainExc()
#%%
if task == "gendata":
    # main_exc.gen_val_v1()
    main_exc._internal_demo(output_dir="nemo_experiments/internal_demo_gen_gs_163k")
    main_exc.v4_gs_val_word_edit(ds_name="gigaspeech", corpus_dir="data/download/GigaSpeech", manifest_filepath="data/parsed/GigaSpeech/gigaspeech_cuts_DEV.speech.jsonl.gz",
                                    output_dir="nemo_experiments/edit_gen_163k/")
    exit()
#%%
    
elif task == "edit":

    main_exc.dataprocessor.prepare_val_dl()
    #%%
    # tb_val_batch = main_exc.dataprocessor.get_val_batch(from_tb=True)
    val_batch = main_exc.dataprocessor.get_val_batch()
    # mix_val_batch = main_exc.dataprocessor.get_val_batch(mix=True)

    #%%
    # tb_val_pred = main_exc.infer.test_batch(tb_val_batch, "nemo_experiments/edit_tb_val", "tb_val")
    val_pred = main_exc.infer.test_batch(val_batch, "nemo_experiments/edit_val", "val")

    #%%
    # _tb_val_batch = tb_val_batch
    _val_batch = val_batch
    # _tb_val_batch = mask_cond(_tb_val_batch)
    _val_batch = main_exc.dataprocessor.mask_cond(_val_batch)
    for i in range(4):
        # Eval.create_plot(_tb_val_batch['cond'][i].T.cpu(), 'frame', 'freq')
        # Eval.create_plot(tb_val_pred[i].T.cpu(), 'frame', 'freq')
        main_exc.eval.create_plot(_val_batch['cond'][i].T.cpu(), 'frame', 'freq')
        main_exc.eval.create_plot(val_pred[i].T.cpu(), 'frame', 'freq')

    #%%
    demo_batch = main_exc.dataprocessor.get_demo_batch(corpus_dir, textgrid_dir)
    ori_demo_batch = main_exc.dataprocessor.get_demo_batch(corpus_dir, textgrid_dir, ori=True)

    #%%
    demo_batch = main_exc.dataprocessor.fix_demo_audio(demo_batch, scale=.65)
    ori_demo_batch = main_exc.dataprocessor.fix_demo_audio(ori_demo_batch, scale=.7)

    #%%
    demo_pred = main_exc.infer.test_batch(demo_batch, "nemo_experiments/edit_demo", "demo-65-")
    ori_demo_pred = main_exc.infer.test_batch(ori_demo_batch, "nemo_experiments/edit_ori_demo", "ori_demo-7-")

    #%%
    main_exc.eval.create_plot(demo_batch['cond'][1].T.cpu(), 'frame', 'freq')
    main_exc.eval.create_plot(ori_demo_batch['cond'][1].T.cpu(), 'frame', 'freq')
    #%%
    main_exc.eval.create_plot(demo_pred[1].T.cpu(), 'frame', 'freq')
    main_exc.eval.create_plot(ori_demo_pred[1].T.cpu(), 'frame', 'freq')

    print(main_exc.model.tokenizer.vocab)
    # exit()



    # %%
