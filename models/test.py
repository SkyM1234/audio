import torch
from newser_model import Ser_Model

model = Ser_Model()


# audio_spec: [batch, 3, 256, 384]
    # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]

batch_size = 2
audio_spec = torch.randn(batch_size, 3, 128, 384)  # [batch, 3, 256, 384]
audio_mfcc = torch.randn(batch_size, 300, 40)      # [batch, 300, 40]
audio_wav = torch.randn(batch_size, 48000)        # [batch, 48000]


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# audio_spec = audio_spec.to(device)
# audio_mfcc = audio_mfcc.to(device)
# audio_wav = audio_wav.to(device)


output = model(audio_spec, audio_mfcc, audio_wav)

# print(output)