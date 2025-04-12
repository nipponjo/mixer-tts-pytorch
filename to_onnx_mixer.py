# %%
import onnx
import torch
import matplotlib.pyplot as plt
from models.mixer_tts.mixer_tts import MixerTTSModel as MixerTTS

n = [384, 128, 80][0]

sds = torch.load(f'./pretrained/mixer_lj_{n}.pth')
if 'net_config' in sds: net_config = sds['net_config']
model = MixerTTS(**net_config)
model.load_state_dict(sds['model'])

# %%

def pred_fun(x: torch.LongTensor, 
             pace: float = 1.0, 
             speaker: int = 0, 
             emotion: int = 0, 
             p_mul: float = 1, 
             p_add: float = 0,
             ):
    def _pitch_trf(pitch_pred, enc_mask_sum, mean, std):
        # print(pitch_pred, enc_mask_sum, mean, std)
        return p_mul*pitch_pred + p_add    
    
    return model.infer(
        x,
        pace = pace,
        pitch_transform = _pitch_trf,            
        speaker = speaker,
        emotion = emotion,
        ).transpose(1,2)

model.forward = pred_fun


# %%

token_ids = torch.LongTensor([[1,2,3]])
pace = torch.FloatTensor([1.])
speaker = torch.IntTensor([0])
emotion = torch.IntTensor([0])
pitch_mul = torch.FloatTensor([1.])
pitch_add = torch.FloatTensor([0.])

test_sample = (token_ids, pace, speaker, emotion, 
               pitch_mul, pitch_add,)

test_output = model(*test_sample)
print(test_output.shape)

# %%

export_16b = False

output_filepath = f'pretrained/mixer_lj_{n}{"_16b" if export_16b else ""}.onnx'

with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=export_16b):
    torch.onnx.export(model.cuda(), 
                      tuple([x.cuda() for x in test_sample]), 
                    output_filepath, 
                    export_params=True,
                    do_constant_folding=True,        
                    verbose=True,          
                    opset_version=11,
                    input_names=['token_ids', 'pace', 
                                 'speaker', 'emotion', 
                                 'pitch_mul', 'pitch_add',                               
                                 ],
                    output_names=['mel_spec'],
                    dynamic_axes={
                        'token_ids': {0: 'batch', 1: 'token'},
                        'mel_spec': {0: 'batch', 2: 'frame'},
                            }
                    )

# %%

# Load the ONNX model
onnxmodel = onnx.load(output_filepath)

# Check that the model is well formed
onnx.checker.check_model(onnxmodel)

# %%

import onnxruntime as ort
import numpy as np
from utils.phonemizer import text_to_ids_en

ort_session = ort.InferenceSession(output_filepath, 
                                   providers=['CPUExecutionProvider'])

# %%

ids_batch = np.array(text_to_ids_en('Hello world!')[None], dtype=np.int64)

mel_spec = ort_session.run(
    None, {
        "token_ids": ids_batch, 
        "pace": np.array([1.], dtype=np.float32),
        "speaker": np.array([0], dtype=np.int32),
        "emotion": np.array([0], dtype=np.int32),
        "pitch_mul": np.array([1], dtype=np.float32),
        "pitch_add": np.array([0], dtype=np.float32),
    },)[0].astype(np.float32)

print(mel_spec.shape)

plt.imshow(mel_spec[0], origin='lower', aspect='auto', interpolation='none')

# %%
