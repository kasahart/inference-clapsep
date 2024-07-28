from inference_model.CLAPSep import CLAPSep
import torch
import librosa
import numpy as np


def load_model(model_config = None, CLAP_path=None, model_checkpoint=None):
    if model_config is None:
        model_config = {"lan_embed_dim": 1024,
            "depths": [1, 1, 1, 1],
            "embed_dim": 128,
            "encoder_embed_dim": 128,
            "phase": False,
            "spec_factor": 8,
            "d_attn": 640,
            "n_masker_layer": 3,
            "conv": False}
    if CLAP_path is None:
        CLAP_path = "inference_model/music_audioset_epoch_15_esc_90.14.pt"
    
    if model_checkpoint is None:
        model_checkpoint = 'inference_model/best_model.ckpt'
        
    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLAPSep(model_config, CLAP_path).to(device)
    ckpt = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def inference(model, audio_file_path: str, text_p: str, audio_file_path_p: str, text_n: str, audio_file_path_n: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # handling queries
    with torch.no_grad():
        embed_pos, embed_neg = torch.chunk(model.clap_model.get_text_embedding([text_p, text_n],
                                                                              use_tensor=True), dim=0, chunks=2)
        embed_pos = torch.zeros_like(embed_pos) if text_p == '' else embed_pos
        embed_neg = torch.zeros_like(embed_neg) if text_n == '' else embed_neg
        embed_pos += (model.clap_model.get_audio_embedding_from_filelist(
            [audio_file_path_p]) if audio_file_path_p is not None else torch.zeros_like(embed_pos))
        embed_neg += (model.clap_model.get_audio_embedding_from_filelist(
            [audio_file_path_n]) if audio_file_path_n is not None else torch.zeros_like(embed_neg))
        
        embed_neg = embed_neg.to(device)
        embed_pos = embed_pos.to(device)



    print(f"Separate audio from [{audio_file_path}] with textual query p: [{text_p}] and n: [{text_n}]")
    model = model.to(device)
    mixture, _ = librosa.load(audio_file_path, sr=32000)

    pad = (320000 - (len(mixture) % 320000))if len(mixture) % 320000 != 0 else 0

    mixture =torch.tensor(np.pad(mixture,(0,pad))).to(device)
    
    max_value = torch.max(torch.abs(mixture))
    if max_value > 1:
        mixture *= 0.9 / max_value
    
    mixture_chunks = torch.chunk(mixture, dim=0, chunks=len(mixture)//320000)
    sep_segments = []
    for chunk in mixture_chunks:
        with torch.no_grad():
            sep_segments.append(model.inference_from_data(chunk.unsqueeze(0), embed_pos, embed_neg))

    sep_segment = torch.concat(sep_segments, dim=1)

    if device == 'cuda':
        sep_segment = sep_segment.cpu()
        
    return 32000, sep_segment.squeeze().numpy()


if __name__ == '__main__':
    model = load_model()
    audio_file_path = "data/audio1.wav"
    text_p = 'A man talking as insects buzz by.'
    audio_file_path_p = None
    text_n = 'Plastic crackling as a bird is singing and chirping.'
    audio_file_path_n = None
    sr, pred = inference(model, audio_file_path, text_p, audio_file_path_p, text_n, audio_file_path_n)
    print(pred.shape)