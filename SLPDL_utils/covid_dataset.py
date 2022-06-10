
import numpy as np
import pandas as pd
import scipy.io.wavfile
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

from frameworks.pytorch.datasets.kaldi_scp_dataset import SCP_Dataset
from frameworks.pytorch.datasets.wrapping_datasets.label_mapping_dataset import LabelListMappingDataset
from frameworks.scipy.mfcc import mfsc, mfsc2mfcc


class COVID_Dataset(Dataset):

    def __init__(self, scp_dataset):
        self.scp_dataset = scp_dataset

    def __getitem__(self, idx):
        key, filename, label = self.scp_dataset[idx]
        sfr, wav = scipy.io.wavfile.read(filename)
        return [key, wav, sfr, label]

    def __len__(self):
        return len(self.scp_dataset)

class COVID_Feat_Dataset(Dataset):
    
    def __init__(self, covid_dataset, feat_extractor, transform=None, target_transform=None):
        self.covid_dataset = covid_dataset
        self.feat_extractor = feat_extractor

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        key, wav, sfr, label = self.covid_dataset[idx]

        y = wav / 32768
        param = self.feat_extractor(y, sfr)

        param = torch.FloatTensor(param)

        if self.transform is not None:
            params = self.transform(params)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return key, param, label
    
    def __len__(self):
        return len(self.covid_dataset)


def build_mel_param_loader(window_size, window_stride, window, normalize, max_len):
    def param_loader(y, sfr):
        y -= y.mean()
        param = mfsc(y, sfr, window_size=window_size, window_stride=window_stride, window=window, normalize=normalize, log=False, n_mels=60, preemCoef=0, melfloor=1.0, n_fft=2048)

        # Add zero padding to make all param with the same dims
        if param.shape[1] < max_len:
            pad = np.zeros((param.shape[0], max_len - param.shape[1]))
            param = np.hstack((pad, param))

        # If exceeds max_len keep last samples
        elif param.shape[1] > max_len:
            param = param[:, -max_len:]

        param = torch.FloatTensor(param)

        return param

    return param_loader

def build_vec_param_loader(processor, max_seconds):
    def param_loader(y, sfr):
        y = y.astype(np.float32)
        y -= y.mean()
        y.resize(max_seconds * sfr)
        y = processor(y, sampling_rate=sfr, return_tensors="np").input_values
        y = y.squeeze(0).astype(np.float32)
        return y
    return param_loader

def get_classes():
    classes = ['neg', 'pos']
    weight = None
    class_to_id = {label: i for i, label in enumerate(classes)}
    return classes, weight, class_to_id

def build_covid_dataset(scp_filepath, param_loader_params, mel_feats=True, basepath='', labels_filepath=None):
    
    base_scp_dataset = SCP_Dataset(scp_filepath, basepath=basepath, labels_filepath=labels_filepath)
    scp_dataset = LabelListMappingDataset(base_scp_dataset, get_classes()[2], modified_indexs=[-1], default_value=-1)
    base_covid_dataset = COVID_Dataset(scp_dataset)
    
    if mel_feats:
        param_loader = build_mel_param_loader(param_loader_params.window_size, param_loader_params.window_stride, param_loader_params.window_type, param_loader_params.normalize, param_loader_params.max_len)
    else:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(param_loader_params.MODEL)
        param_loader = build_vec_param_loader(processor, param_loader_params.max_seconds)

    covid_dataset = COVID_Feat_Dataset(base_covid_dataset, param_loader)
    return covid_dataset

def build_covid_collate(device=None):
    # Maybe no needed new collate
    #device = device or torch.device('cpu')

    def collate_fn(batch):
        key, params, labels = zip(*batch)

        params = torch.stack(params)
        #params.to(device)

        labels = torch.stack( [torch.IntTensor(l) for l in labels] )
        #labels.to(device)

        return key, params, labels
    return collate_fn
