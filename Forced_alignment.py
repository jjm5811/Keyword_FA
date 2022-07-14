import os
from dataclasses import dataclass
import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython
import pdb
import glob

def get_trellis(emission, tokens, blank_id=0):
  num_frame = emission.size(0)
  num_tokens = len(tokens)

  # Trellis has extra diemsions for both time axis and tokens.
  # The extra dim for tokens represents <SoS> (start-of-sentence)
  # The extra dim for time axis is for simplification of the code. 
  trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
  trellis[:, 0] = 0
  for t in range(num_frame):
    trellis[t+1, 1:] = torch.maximum(
        # Score for staying at the same token
        trellis[t, 1:] + emission[t, blank_id],
        # Score for changing to the next token
        trellis[t, :-1] + emission[t, tokens],
    )
  return trellis

''' Find the most likely path '''
@dataclass
class Point:
  token_index: int
  time_index: int
  score: float


def backtrack(trellis, emission, tokens, blank_id=0):
  j = trellis.size(1) - 1
  t_start = torch.argmax(trellis[:, j]).item()

  path = []
  for t in range(t_start, 0, -1):
    # 1. Figure out if the current position was stay or change
    stayed = trellis[t-1, j] + emission[t-1, blank_id]
    changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

    # 2. Store the path with frame-wise probability.
    prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()

    # Return token index and time index in non-trellis coordinate.
    path.append(Point(j-1, t-1, prob))

    # 3. Update the token
    if changed > stayed:
      j -= 1
      if j == 0:
        break
  else:
    return None
    # raise ValueError('Failed to align')

  return path[::-1]


''' Merge the labels '''
@dataclass
class Segment:
  label: str
  start: int
  end: int
  score: float

  def __repr__(self):
    return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

  @property
  def length(self):
    return self.end - self.start

def merge_repeats(path):
  i1, i2 = 0, 0
  segments = []
  while i1 < len(path):
    while i2 < len(path) and path[i1].token_index == path[i2].token_index:
      i2 += 1
    score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
    segments.append(Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
    i1 = i2
  return segments


''' Merge words '''
def merge_words(segments, separator=' '):
  words = []
  i1, i2 = 0, 0
  while i1 < len(segments):
    if i2 >= len(segments) or segments[i2].label == separator:
      if i1 != i2:
        segs = segments[i1:i2]
        word = ''.join([seg.label for seg in segs])
        score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
        words.append(Segment(word, segments[i1].start, segments[i2-1].end, score))
      i1 = i2 + 1
      i2 = i1
    else:
      i2 += 1
  return words


''' Save file '''
def display_segment(i):
  ratio = waveform.size(1) / (trellis.size(0) - 1)
  margin = 0.03
  ratio0 = ratio*(1-margin)
  ratio1 = ratio*(1+margin)

  word = word_segments[i]
  x0 = int(ratio0 * word.start)
  x1 = int(ratio1 * word.end)
  directory = f"/mnt/scratch/datasets/words/{word.label}"  ## You need to change here !!! (save path)
  if not os.path.exists(directory):
            os.makedirs(directory)
  filename = f"{directory}/{word.label}_{file_name}_{i}.wav"

  torchaudio.save(filename, waveform[:, x0:x1], 16000) # x0 = 7048 x1 = 8009
  print(f"{word.label} ({word.score:.2f}): {x0 / 16000:.3f} - {x1 / 16000:.3f} sec")


#######################################################
'''-------------- Main ---------------'''
#######################################################

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' Generate frame-wise label probability '''
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()

dictionary  = {c: i for i, c in enumerate(labels)}
del(dictionary['<unk>'], dictionary['<s>'])
dictionary['*'] = 3 
dictionary[' '] = 0

''' Generate alignment probability '''
## You need to change here !!! (load path)
## I used all of dataset in Librispeech
folder_name = 'train-960' # 'train-960', 'dev-other', 'dev-clean', 'test-other', test-clean'
txt_files = glob.glob('/mnt/scratch/datasets/LibriSpeech/' + folder_name + '/*/*/*.txt', recursive=True)

for i in range(len(txt_files)):
    f = open(txt_files[i], 'r')
    try:
      lines = f.readlines()
    except:
      print('error in read file')
    for line in lines:
        file_name = line.split(' ')[0]
        origin_line = line[len(file_name)+1:]

        folder1, folder2, _ = file_name.split('-')
        SPEECH_FILE = '/mnt/scratch/datasets/LibriSpeech/'+ folder_name +'/'+ folder1 +'/'+ folder2 +'/'+ file_name + '.flac'
        
        with torch.inference_mode():
            try:
                waveform, _ = torchaudio.load(SPEECH_FILE)
            except:
                continue
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)
            
        ## Remove useless characters
        emission = emissions[0].cpu().detach()
        num_char = "0123456789"
        for num in num_char:
            changed_line = origin_line[:-1].replace(num, '*')

        changed_line = [x for x in changed_line if x in dictionary]
        changed_line = ''.join(changed_line)

        len_word = len(changed_line.split(' ')) 
    
        transcript = changed_line 

        #align
        tokens = [dictionary[c] for c in transcript]

        trellis = get_trellis(emission, tokens)

        path = backtrack(trellis, emission, tokens)
        if path == None:
            continue

        segments = merge_repeats(path)

        word_segments = merge_words(segments)

        for i in range(len_word):
            display_segment(i)

