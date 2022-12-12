# %matplotlib inline

import os
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import editdistance
from evaluate import load
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
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H

model = bundle.get_model().to(device)

''' Loading data'''
f = open('./make_filelist/files_margin_1s.txt')  ## You need to change HERE!
lines = f.readlines()

num_correct = 0
num_incorrect = 0
num_one_char = 0
num_save = 0
TO_BE_UESD_LINES = lines #lines[::900]
fw = open("./words_filter/files_filtered_1s_cer20.txt", 'w')

for line in tqdm(TO_BE_UESD_LINES):
    SPEECH_FILE = line.split('|')[1][:-1]
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to(device)
    
    word_label =line.split('|')[0]
    word_label = word_label.replace('\'', '') #따옴표 제거

    # 한 글자 데이터 제외
    if len(word_label) == 1 :
        num_one_char += 1
        continue

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    ''' Main ''' 
    with torch.inference_mode():
        emission, _ = model(waveform)

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])
    words = transcript.split('|') #Prediction words

    ''' Save '''
    directory = f"/mnt/scratch/datasets/words_filter_1s_cer20/{word_label}" ## You need to change HERE!
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

    ## words_filter(CER)
    detect_keyword = []

    for word in words_processed:
        find_keyword = False
        dist = editdistance.eval(word, word_label)
        length = len(word_label)
        cer = dist/length
        
        if cer <= 0.2: 
            num_correct += 1
            find_keyword = True

        detect_keyword.append(find_keyword)
    
    ## check audio quality
    if True not in detect_keyword:
        num_incorrect += 1
        
    num_true = detect_keyword.count(True)

    if num_true > 1 :
        num_correct -= (num_true-1)

    ## Save correct file
    elif num_true == 1:
        torchaudio.save(filename, waveform , 16000) 
        fw.write(word_label + ' ' + word_label + '/' + file_name + '\n')
        num_save += 1

print('num_correct: ' + str(num_correct) + '/' + str(len(TO_BE_UESD_LINES)))
print('num_incorrect: ' + str(num_incorrect) + '/' + str(len(TO_BE_UESD_LINES)))
print('num_one_char: ' + str(num_one_char) + '/' + str(len(TO_BE_UESD_LINES)))
print('num_save: ' + str(num_save) + '/' + str(len(TO_BE_UESD_LINES)))
print('Total number of data: ' + str(num_correct + num_incorrect + num_one_char))

f.close()
fw.close()