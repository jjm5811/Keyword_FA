# %matplotlib inline

import os
from dataclasses import dataclass

import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
# import IPython

matplotlib.rcParams['figure.figsize'] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_URL = 'https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
SPEECH_FILE = '_assets/speech.wav'

if not os.path.exists(SPEECH_FILE):
  os.makedirs('_assets', exist_ok=True)
  with open(SPEECH_FILE, 'wb') as file:
    file.write(requests.get(SPEECH_URL).content)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
  waveform, _ = torchaudio.load(SPEECH_FILE)
  emissions, _ = model(waveform.to(device))
  emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

print(labels)
plt.imshow(emission.T)
plt.colorbar()
plt.title("Frame-wise class probability")
plt.xlabel("Time")
plt.ylabel("Labels")
plt.show()