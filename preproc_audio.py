# %%
import os
import glob
import torchaudio
from utils.audio import fft_filter, rms_normalize

# This script will:
# 1. remove frequency content below 60Hz
# 2. RMS normalize the signal energy at -27dB
# These operations could also be performed at training time,
# but here we do them as preprocessing to save some time

dir_orig = 'I:/tts/english/LJSpeech-1.1/wavs'
dir_target = 'I:/tts/english/LJSpeech-1.1/wavs_proc'
os.makedirs(dir_target, exist_ok=True)

filepaths = glob.glob(f"{dir_orig}/*.wav")

# %%

for filepath in filepaths:
    audio, sr = torchaudio.load(filepath)
    filename = os.path.basename(filepath)
    
    audio = fft_filter(audio, sr, 60)
    audio = rms_normalize(audio, -27)
    
    torchaudio.save(os.path.join(dir_target, filename),
                    audio, sr)
    


