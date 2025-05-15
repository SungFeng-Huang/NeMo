import torch
import librosa
from nemo.collections.tts.models import AudioCodecModel


codec_path = "/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Low_Frame-rate_Speech_Codec++_nemo_last_checkpoint/ml-model-INTERSPEECH_2025_abblations_12.5Hz_8_codebooks_2016_codes_pad_fix_enc_non_causal_dec_causal_/checkpoints/Low_Frame-rate_Speech_Codec++.nemo"
path_to_input_audio = "/lustre/fsw/portfolios/nvr/users/sungfengh/datasets/TechOrange/mount/src/NeMo/ASR/TechOrange_tp1/241209/preprocessed/trial/processed_audio_segment_50_39.wav"


# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load model
codec_model = AudioCodecModel.restore_from(restore_path=codec_path, map_location="cpu", strict=True).eval().to(device)
# freeze codec to ensure no weight update
codec_model.freeze()


# load a dummy audio audio in 22khz (codec_model.sample_rate)
input_audio, _ = librosa.load(path_to_input_audio, sr=codec_model.sample_rate)


# audio to tensor
input_audio = torch.from_numpy(input_audio).unsqueeze(dim=0).to(device)
# audio len is the number of frames in the time dimension
input_audio_len = torch.tensor([input_audio.size(1)]).to(device)


# get tokens and dequantized latent using autocast to ensures precision 32 (codec only works on this precision for now)
with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
    # get audio tokens
    gen_tokens, gen_tokens_lens = codec_model.encode(
        audio=input_audio, audio_len=input_audio_len
    )
    gen_audio, _ = codec_model.decode(tokens=gen_tokens, tokens_len=gen_tokens_lens) # shape [B, T]
