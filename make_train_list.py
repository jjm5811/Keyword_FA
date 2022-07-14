import sys, time, argparse
import re
import os
from pathlib import Path
import glob
import random,pdb
import pickle

random.seed(0)

parser = argparse.ArgumentParser(description = "Data Divider")

parser.add_argument('--dataset_path', type=str, default='/mnt/scratch/datasets/', help='parent path of dataset directory')
parser.add_argument('--dataset_name', type=str, default='words_filtered', help='dataset (directory) name')
parser.add_argument('--spec_path', type=str, default='', help='specific path') ## /TEDLIUM_v3/legacy_asr

args = parser.parse_args()

data_path = args.dataset_path + args.dataset_name + args.spec_path

root_path = Path(data_path)

with open('words_list.pkl', 'rb') as f:
    words_list = pickle.load(f)

num_test = 10 # number of test keywords 

num_words = 1000
cut_words = 17

num_audios = 1000 # number of utterances per one keyword class

train_list = words_list[cut_words:cut_words+num_words]
test_list = words_list[cut_words+num_words:cut_words+num_words+num_test]

train_words = []
test_words = []
for i in range(len(train_list)):
    train_words.append(train_list[i][0])

for i in range(len(test_list)):
    test_words.append(test_list[i][0])

DATA_CONFIG = {'train' : train_words, 'test' : test_words}

## rand_sample SHOULD BE UNDER 190 (i.e. ~189) because of the number of positive samples ##
rand_sample = 30 # number of pos & neg samples per one keyword 
rand_select = 10 # number of randomly selected anchor wav

def make_train_list(data_config=DATA_CONFIG, data_path=data_path):
    f_train = open(root_path / 'train_list.txt', 'w')
    for keyword in data_config['train']:
        word_direc = root_path / keyword
        wav_f_list = os.listdir(word_direc)
        num_wav = len(wav_f_list)
        if num_wav < num_audios:
            rep = num_audios // num_wav + 1
            f_list = wav_f_list * rep
            f_list = f_list[:num_audios]
            for wav_f in f_list:
                f_train.write(keyword + ' ')
                f_train.write(keyword + '/' + wav_f + '\n')
        else:
            f_list = wav_f_list[:num_audios]
            for wav_f in f_list:
                f_train.write(keyword + ' ')
                f_train.write(keyword + '/' + wav_f + '\n')
    f_train.close()

def make_test_list(data_config=DATA_CONFIG, data_path=data_path):
    f_test   = open(os.path.join(data_path, 'test_list.txt'), 'w')
    keys     = data_config['test']
    f_dict   = {}
    anc_dict = {}

    for d in os.listdir(data_path):
        if d in keys:
            _wav_f_list = os.listdir(os.path.join(data_path, d))
            wav_f_list  = [d+'/'+item for item in _wav_f_list]
            rand_f_list = random.sample(wav_f_list, rand_select)

            f_dict[d]   = [item for item in wav_f_list if item not in rand_f_list]
            anc_dict[d] = rand_f_list

    for key in keys:
        pos_samples = []
        neg_samples = []

        pos_samples = random.sample(f_dict[key], rand_sample * (num_test - 1))
        neg_samples = [random.sample(v, rand_sample) for k, v in f_dict.items() if k != key]
        neg_samples = [item for sublist in neg_samples for item in sublist]
        neg_samples = random.sample(neg_samples, len(neg_samples))

        pos_pairs = []
        neg_pairs = []
        for anc_f in anc_dict[key]:
            pos_pairs.append([('1', anc_f, pos_sample) for pos_sample in pos_samples])
            neg_pairs.append([('0', anc_f, neg_sample) for neg_sample in neg_samples])

        pos_pairs = [item for sublist in pos_pairs for item in sublist]
        neg_pairs = [item for sublist in neg_pairs for item in sublist]

        whole_pairs = [x for y in zip(pos_pairs, neg_pairs) for x in y]

        for pair in whole_pairs:
            f_test.write(pair[0]+' '+pair[1]+' '+pair[2]+'\n')

    f_test.close()
 
make_train_list();
make_test_list();