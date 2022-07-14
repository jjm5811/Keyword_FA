# %matplotlib inline

import os
import pdb, glob

''' Loading data'''
SPEECH_FILE_list = glob.glob('/mnt/scratch/datasets/words/*/*.wav', recursive=True)

f = open("./files.txt", 'w')

for SPEECH_FILE in SPEECH_FILE_list:
    label = SPEECH_FILE.split('/')[5]
    f.write(label + '|' + SPEECH_FILE + '\n')


f.close()