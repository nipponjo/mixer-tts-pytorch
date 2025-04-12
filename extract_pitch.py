# %%
import os
import librosa
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.audio import MelSpectrogram
from utils import write_lines_to_file, make_dataset_from_subdirs
mel_trf = MelSpectrogram()


audio_dir = "I:/tts/english/LJSpeech-1.1"
pitch_dir = "./data/lj_pitches"

audio_filepaths = make_dataset_from_subdirs(audio_dir)

print("Found", len(audio_filepaths), "audio files")


# %% extract pitch (f0) values


for i, audio_path in tqdm(enumerate(audio_filepaths), total=len(audio_filepaths)):
    wav, sr = librosa.load(audio_path, sr=mel_trf.sample_rate)

    audio_relpath = os.path.relpath(audio_path, audio_dir)
    pitch_filepath = os.path.join(pitch_dir, audio_relpath) + '.pt'
    
    if os.path.exists(pitch_filepath): continue
    pitch_dirname = os.path.dirname(pitch_filepath)
    if not os.path.exists(pitch_dirname): os.makedirs(pitch_dirname)
    
    mel_spec = mel_trf(torch.tensor(wav)[None])[0] # [mel_bands, T]

    # estimate pitch
    pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
        wav, sr=mel_trf.sample_rate,
        fmin=60, 
        fmax=2000,
        frame_length=mel_trf.win_length,
        hop_length=mel_trf.hop_length)

    pitch_mel = np.where(np.isnan(pitch_mel), 0., pitch_mel) # set nan to zero
    pitch_mel = torch.from_numpy(pitch_mel)
    pitch_mel = F.pad(pitch_mel, (0, mel_spec.size(1) - pitch_mel.size(0))) # pad to mel length

    torch.save(pitch_mel, pitch_filepath)


# %% calculate pitch mean and std


pitch_filepaths = make_dataset_from_subdirs(pitch_dir, ('.wav.pt'))

rmean = 0
rvar = 0
ndata = 0

for pitch_filepath in tqdm(pitch_filepaths):
    pitch_mel = torch.load(pitch_filepath)
    
    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    
    pitch_mel_ = pitch_mel[pitch_mel > 1]
    p_mean = np.mean(pitch_mel_)
    p_var = np.var(pitch_mel_)
    p_len = len(pitch_mel_)

    rvar = ((ndata-1)*rvar + (p_len-1)*p_var) / (ndata + p_len - 1) + \
            ndata*p_len*(p_mean - rmean)**2 / ((ndata + p_len)*(ndata + p_len - 1))
    
    rmean = (p_len*p_mean + ndata*rmean) / (p_len + ndata)

    ndata += p_len

mean, std = rmean, np.sqrt(rvar)
print('mean ', mean)
print('std ', std)

textfile_path = './data/mean_std.txt'
write_lines_to_file(path=textfile_path, 
                    lines=[f"mean: {mean}", 
                           f"std: {std}"])
print(f"Saved mean and std @ {textfile_path}")


# %%
