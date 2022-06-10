
# Kaldi defines .scp as "file_id filepath/filename:integer" rows (:integer is optional, currently it will be omited)
# Here the dataset allows the addition of a basepath: basepath/filepath/filename
# Here the dataset allows to load labels from a file with "file_id label" rows
# TODO: manage the integer when using ark files


import os
import torch
from torch.utils.data import Dataset


class SCP_Dataset(Dataset):
    # TODO: Improve loading and add all the possibilities of SCP

    def __init__(self, scp_filepath, basepath='', labels_filepath=None):
        key_to_word = dict()
        key_to_wav = dict()

        with open(scp_filepath, 'rt') as wav_scp:
            for line in wav_scp:
                key, wav = line.strip().split(' ', 1)
                key_to_wav[key] = f"{basepath}/{wav.split(':', 1)[0]}"
                key_to_word[key] = None # default
        
        if labels_filepath is not None:
            with open(labels_filepath, 'rt') as text:
                for line in text:
                    key, word = line.strip().split(' ', 1)
                    key_to_word[key] = word

        self.list_of_items = [[key, wav_command, key_to_word[key]] for key, wav_command in key_to_wav.items()]
    
    def __getitem__(self, idx):
        return self.list_of_items[idx]
    
    def __len__(self):
        return len(self.list_of_items)
