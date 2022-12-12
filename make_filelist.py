# %matplotlib inline

import pdb, glob
from tqdm import tqdm

''' Loading data'''
SPEECH_FILE_list = glob.glob('/mnt/work4/datasets/keyword/words_margin_1s/*/*.wav', recursive=True)

f = open("./make_filelist/files_margin_1s.txt", 'w')

for SPEECH_FILE in tqdm(SPEECH_FILE_list):
    label = SPEECH_FILE.split('/')[-2]
    f.write(label + '|' + SPEECH_FILE + '\n')

f.close()