# mixer-tts-pytorch

[[Samples]](https://nipponjo.github.io/tts-mixer-samples/)

This repo contains an implementation of the Mixer-TTS model ([https://arxiv.org/abs/2110.03584](https://arxiv.org/abs/2110.03584)).

Pre-trained weights are available for the [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset. 
The channel dimensions of the convolutions inside the model were chosen as 384, 128 and 80, resulting in models with 20.6M, 3.17M and 1.74M parameters.

The pre-trained models take IPA symbols as input. Please refer to [here](https://bootphon.github.io/phonemizer/install.html) to install `phonemizer` and the `espeak-ng` backend.
An simple patch-based discriminator was used in training to generate more natural mel spectrograms.

Audio samples are available [here](https://nipponjo.github.io/tts-mixer-samples/).

Pre-trained models:

All (3) checkpoint files can be downloaded by running: `python download_files.py`

|Dataset|dim|params|name|link|
|-------|---|------|-----|---|
|LJSpeech|80|1.74M|mixer_lj_80|[link](https://drive.google.com/file/d/1YTiA6S3okiuX-_AttUhJNVgiPzVYAyjv/view?usp=sharing)|
|LJSpeech|128|3.17M|mixer_lj_128|[link](https://drive.google.com/file/d/1wVvOyaBLxqrKAssXmEYG9mszZsqEaX5R/view?usp=sharing)|
|LJSpeech|384|20.6M|mixer_lj_384|[link](https://drive.google.com/file/d/16Rq99ZmXVfiDE_nsxmUBzF3XKEOUh5wx/view?usp=sharing)|


The pre-trained models output the 80-channel mel spectrogram version first proposed by the [HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder.


References:

The model was taken out of NVIDIA's [NeMo](https://github.com/NVIDIA/NeMo) framework in order to make it easier to modify and have fewer dependencies. An energy embedding and optional speaker and emotion embeddings have been added.

[Mixer-TTS in NeMo](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/models/mixer_tts.py)

Paper:
```
@article{Tatanov2021MixerTTSNF,
  title={Mixer-TTS: Non-Autoregressive, Fast and Compact Text-to-Speech Model Conditioned on Language Model Embeddings},
  author={Oktai Tatanov and Stanislav Beliaev and Boris Ginsburg},
  journal={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2021},
  pages={7482-7486},
}
```