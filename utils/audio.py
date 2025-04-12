import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


def rms_normalize(audio: torch.Tensor, 
                  target_dB: float = -27):
    """
    Normalize the RMS of an audio tensor to a specific dB level.
    
    Args:
        audio (Tensor): The input audio waveform (1D or 2D tensor for mono/stereo audio).
        target_dB (float): The target RMS level in dB.
        
    Returns:
        Tensor: The RMS-normalized audio.
    """
    # Calculate the RMS value of the audio
    rms = torch.sqrt(torch.mean(audio ** 2))

    # Convert target dB to a linear scale
    target_rms = 10 ** (target_dB / 20)

    # Compute the gain factor
    gain = target_rms / (rms + 1e-8)  # Avoid division by zero

    # Apply gain to the audio signal
    normalized_audio = audio * gain

    return normalized_audio

def fft_filter(wave: torch.Tensor, 
               sr: int, 
               cutoff:float=60):    
    wave_fft = torch.fft.rfft(wave)
    ns = wave_fft.size(-1) / (sr/2)
    nb = int(cutoff*ns)
    if nb < 1: return wave
    wave_fft[...,:nb] = 0
    wave_filt = torch.fft.irfft(wave_fft)
    return wave_filt


class MelSpectrogram(nn.Module):
    def __init__(self, 
                 sample_rate=22050,
                 n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 n_mels=80,
                 f_min=0,
                 f_max=8000.0,
                 norm='slaney',
                 center=False
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        self.pad_length = int((n_fft - hop_length)/2)

        mel_basis = torch.Tensor(
            librosa_mel_fn(sr=sample_rate,
                           n_fft=n_fft, 
                           n_mels=n_mels,
                           fmin=f_min, 
                           fmax=f_max, 
                           norm=norm,
                           ))
        window_fn = torch.hann_window(win_length)
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window_fn', window_fn)

    def forward(self, x):
        x_pad = torch.nn.functional.pad(
            x, (self.pad_length, self.pad_length), mode='reflect')
        spec_ta = torch.stft(x_pad, self.n_fft,
                             self.hop_length,
                             self.win_length,
                             self.window_fn,
                             center=self.center,
                             return_complex=False)
        spec_ta = torch.sqrt(spec_ta.pow(2).sum(-1) + 1e-9)
        mel_ta2 = torch.matmul(self.mel_basis, spec_ta)
        return mel_ta2
