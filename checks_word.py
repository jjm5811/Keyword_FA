# %matplotlib inline

import os
import pdb, glob
import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import editdistance
from tqdm import tqdm

''' CTC Decoder '''
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

''' Preparataion'''
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Creating Pipeline'''
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

model = bundle.get_model().to(device)

''' Loading data'''
f = open('./files.txt')
lines = f.readlines()

num_correct = 0

for line in tqdm(lines):
    SPEECH_FILE = line.split('|')[1][:-1]
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to(device)
    
    word_label =line.split('|')[0]
    word_label = word_label.replace('\'', '') #따옴표 제거

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    ''' Main ''' 
    with torch.inference_mode():
        emission, _ = model(waveform)

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])
    words = transcript.split('|') #Prediction words

    ''' Save '''
    directory = f"/mnt/scratch/datasets/words_filtered/{word_label}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = line.split('/')[-1][:-1]
    filename = f"{directory}/{file_name}"
    waveform = waveform.cpu().detach()
    
    ## Pre-processing
    words_processed = []
    for word in words:  
        word = word.replace('\'', '')
        words_processed.append(word)

    for word in words_processed:
        error = editdistance.eval(word, word_label)

        if len(word) > 2 and error < 4 :
            torchaudio.save(filename, waveform , 16000) 
            num_correct += 1

        if len(word) < 3 and error < 2 : 
            torchaudio.save(filename, waveform , 16000) 
            num_correct += 1

    # transcript = '|'.join(words_processed)
    # print('Pred: ' + transcript)
    # print('GT: ' + word_label)'

total_words = len(lines)
print('num_correct: ' + str(num_correct) + '/' + str(total_words))
