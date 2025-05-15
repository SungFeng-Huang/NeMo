import torch
import librosa
from nemo.collections.audio.models.enhancement import FlowMatchingAudioToAudioModel
from scripts.flow_matching_codec_decoder.debug_wrapper import setup_debugging


fm_path = "/lustre/fsw/portfolios/convai/users/ajukic/jobs/results/GenSE_Foundation_updated/flow_matching_24layer_TransformerUNet_ssl_pretraining_librilight_Tmask70_max_steps600000_nodes4_gpus8/checkpoints/flow_matching_24layer_TransformerUNet_ssl_pretraining_librilight_Tmask70_max_steps600000_nodes4_gpus8--val_pesq=1.9573-epoch=23-EMA.nemo"
path_to_input_audio = "/lustre/fsw/portfolios/nvr/users/sungfengh/datasets/TechOrange/mount/src/NeMo/ASR/TechOrange_tp1/241209/preprocessed/trial/processed_audio_segment_50_39.wav"


if __name__ == "__main__":
    # Set up debugpy
    setup_debugging()

    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    fm_model = FlowMatchingAudioToAudioModel.restore_from(restore_path=fm_path, map_location="cpu", strict=True).eval().to(device)
    # freeze codec to ensure no weight update
    fm_model.freeze()

    # load a dummy audio audio in 22khz (fm_model.sample_rate)
    input_audio, _ = librosa.load(path_to_input_audio, sr=fm_model.sample_rate)
    # input_audio, _ = librosa.load(path_to_input_audio, sr=24000)

    # audio to tensor
    input_audio = torch.from_numpy(input_audio).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    # audio len is the number of frames in the time dimension
    input_audio_len = torch.tensor([input_audio.size(2)]).to(device)


    # get tokens and dequantized latent using autocast to ensures precision 32 (codec only works on this precision for now)
    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
        # get audio tokens
        output, output_length = fm_model.forward_eval(
            input_signal=input_audio, input_length=input_audio_len
        )
        # gen_audio, _ = fm_model.decode(tokens=gen_tokens, tokens_len=gen_tokens_lens) # shape [B, T]