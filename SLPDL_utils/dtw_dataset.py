
import numpy as np
import pandas as pd
import scipy.io.wavfile
import torch
from torch.utils.data import Dataset

from frameworks.pytorch.datasets.csv_dataset import CSV_Dataset
from frameworks.scipy.mfcc import mfsc, mfsc2mfcc


FILE = 'wav'
TEXT = 'text'
SPEAKER = 'speaker'

def get_train_names() : return [FILE, TEXT, SPEAKER]
def get_test_names() : return [FILE]

class DTW_Dataset(Dataset):

    def __init__(self, list_path, data_root='', names=None, type_='train'):
        names = names or (get_train_names() if type_ != 'test' else get_test_names())
        self.table_of_pathes = pd.read_csv(list_path, header=None, index_col=False, names=names)

    def __getitem__(self, idx):
        rows = self.table_of_pathes.iloc[idx]
        sfr, wav = scipy.io.wavfile.read(rows[FILE])
        return wav, sfr, *rows.to_list()
    
    def __len__(self):
        return len(self.table_of_pathes.index)

class DTW_MFCC_Dataset(Dataset):
    # This is half ugly, it allows to use GPU with only mfcc (less memory) but it is too complex
    def __init__(self, dtw_dataset, mfsc_funct=None, return_wav=False):
        self.mfsc_funct = mfsc_funct or mfsc
        self.dtw_dataset = dtw_dataset
        self.return_wav = return_wav

    def __getitem__(self, idx):
        data = self.dtw_dataset[idx]
        wav, sfr = data[:2]

        y = wav / 32768
        S = self.mfsc_funct(y, sfr)

        # Compute the mel spectrogram
        M = mfsc2mfcc(S)
        
        # Move the temporal dimension to the first index
        M = M.T
        
        # DM = delta(M)
        # M = np.hstack((M, DM))
        if self.return_wav : return M.astype(np.float32), y, sfr, *data[2:]
        return M.astype(np.float32), sfr, *data[2:]
    
    def __len__(self):
        return len(self.dtw_dataset)


def build_dtw_collate(use_torch=False, device=None, text_labels=None, speaker_labels=None):
    # This is ugly
    device = device or torch.device('cpu')

    def collate_fn(batch):
        if torch == False or text_labels is None or speaker_labels is None : return tuple(zip(*batch))

        mfcc_list = []
        mfcc = None
        inputs = []
        sfr_list = []
        text_targets = []
        text = None
        speaker_targets = []
        speaker = None
        for data in batch:
            if   len(data) == 6 :   mfcc, wav, sfr, filename, text, speaker = data
            elif len(data) == 4 :   mfcc, wav, sfr, filename                = data
            elif len(data) == 3 :   wav, sfr, filename                      = data # This option works for both, "no mfcc & yes wav" and "no wav & yes mfcc"
            elif len(data) == 5 :   wav, sfr, filename, text, speaker       = data

            try:
                text_idx = text_labels.index(text)
            except IndexError:
                text_idx = -1
            
            try:
                speaker_idx = speaker_labels.index(speaker)
            except IndexError:
                speaker_idx = -1

            if mfcc is not None : mfcc_list.append(torch.FloatTensor(mfcc))
            inputs.append(torch.FloatTensor(wav))
            sfr_list.append(torch.IntTensor(srf))
            if text is not None     : text_targets.append(torch.IntTensor(text_idx))
            if speaker is not None  : speaker_targets.append(torch.IntTensor(speaker_idx))
        
        inputs = torch.stack(inputs)
        inputs = inputs.to(device)
        sfr_list = torch.stack(sfr_list)
        sfr_list = sfr_list.to(device)
        
        return_ = [inputs, sfr_list, filename]

        if text is not None:
            text_targets = torch.stack(text_targets)
            text_targets = text_targets.to(device)

            return_.append(text_targets)
        
        if speaker is not None:
            speaker_targets = torch.stack(speaker_targets)
            speaker_targets = speaker_targets.to(device)

            return_.append(speaker_targets)

        if mfcc is not None:
            mfcc_list = torch.stack(mfcc_list)
            mfcc_list = mfcc_list.to(device)

            return_ = [mfcc_list] + return_

        return tuple(return_)
    
    return collate_fn
