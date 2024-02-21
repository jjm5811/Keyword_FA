# LibriSpeech Keyword (LSK) dataset construction

1. run Forced_aligment.py
(Please modify lines 118, 156-157 to match your directory.)
-> Create words directory.

2. run make_filelist.py
(File list in words directory)
-> Create files.txt file.

3. run words_filter.py
(Please modify the path in lines 43, 51, and 77.)
-> Create words_filtered dataset.

## Acknowledgements

https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
