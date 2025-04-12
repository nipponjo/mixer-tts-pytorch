# %%
import torch
from vocoder.vocos.pretrained import Vocos
from models import MixerTTSModel
from utils.phonemizer import text_to_ids_en
import sounddevice as sd

# %%

ckpt = torch.load('./pretrained/mixer_lj_128.pth')

mel_model = MixerTTSModel(**ckpt['net_config'])
mel_model.load_state_dict(ckpt['model'])
mel_model.eval();

# %%

# 22.05kHz: https://huggingface.co/BSC-LT/vocos-mel-22khz
# 44.1kHz:  https://huggingface.co/patriotyk/vocos-mel-hifigan-compat-44100khz

sample_rate = [22050, 44100][0]

if sample_rate == 22050:
    vocos = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")
elif sample_rate == 44100:
    vocos = Vocos.from_pretrained("patriotyk/vocos-mel-hifigan-compat-44100khz")


# %%

text = "Hello World!"

token_ids = text_to_ids_en(text)
mel_spec = mel_model.infer(text=token_ids[None], 
                           text_len=torch.LongTensor([len(token_ids)]))
wave = vocos.decode_mel(mel_spec.transpose(1,2), denoise=0.003)
wave = wave / wave.abs().max()

sd.play(wave[0], sample_rate)

# %%
