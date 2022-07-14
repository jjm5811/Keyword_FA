import os, pdb
import operator
import pickle

CHECK_WORDS = 1027
path = '/mnt/scratch/datasets/words_filtered/'

folder_list = os.listdir(path) 
num_folder = len(folder_list)
# print(num_folder)

num_words = {}
delete_list={}

for folder in folder_list:
    folder_path = path + folder 
    if os.path.isdir(folder_path):
        file_list = os.listdir(folder_path)
    num_file = len(file_list)
    num_words[folder] = num_file

    ''' Remove empty folders '''
    # if num_file == 0:
    #     delete_list[folder] = num_file
    #     os.rmdir(path+folder)

# operator.itemgetter 
num_words_sorted = sorted(num_words.items(), key=operator.itemgetter(1), reverse=True) 

num_total_words = 0
for i in range(CHECK_WORDS):
    num_total_words += num_words_sorted[i][1]

print(num_words_sorted[:CHECK_WORDS])
print(num_total_words)

with open('words_list.pkl', 'wb') as f:
    pickle.dump(num_words_sorted,f)

