
# TODO: remake the list without random shuffling

from datetime import datetime
import numpy as np
import pathlib
import torch
from torch.utils.data import ConcatDataset, DataLoader

import warnings
from numba.core.errors import NumbaWarning
warnings.simplefilter("ignore", category=NumbaWarning)

from frameworks.scipy.dtw import dtw
from frameworks.scipy.mfcc import mfsc, mfsc2mfcc
from SLPDL_utils.dtw_dataset import DTW_Memory_Dataset as DTW_Dataset
from SLPDL_utils.dtw_dataset import DTW_Memory_MFCC_Dataset as DTW_MFCC_Dataset
from SLPDL_utils.dtw_dataset import build_dtw_collate


def predict(test_dataloader, ref_dataloader=None, same_spk=False, return_targets=False, verbose=False):
    # the dataloaders have to be "DTW_MFCC_Dataset with return_wav=True" and "batch_size=1"; TODO: batch_size > 1 

    assert test_dataloader.batch_size == 1
    if ref_dataloader is None : ref_dataloader = test_dataloader
        
    pred_text = []
    pred_speaker = []
    targ_text = []
    targ_speaker = []
    unknow_filename = []
    for i, test_data in enumerate(test_dataloader):
        if same_spk == False or return_targets == True:
            test_mfcc, test_wav, test_sfr, test_filename, test_text, test_speaker = tuple(zip(*test_data))[0]
        else:
            test_mfcc, test_wav, test_sfr, test_filename = tuple(zip(*test_data))[0]

        mincost = np.inf
        minref_text = None
        minref_speaker = None
        for j, ref_data in enumerate(ref_dataloader):
            ref_mfcc, ref_wav, ref_sfr, ref_filename, ref_text, ref_speaker = tuple(zip(*ref_data))[0]

            if same_spk == False and test_speaker == ref_speaker:
                # Do not compare with refrence recordings of the same speaker
                continue

            if test_filename != ref_filename:
                distance = dtw(test_mfcc, ref_mfcc)

                if distance < mincost:
                    mincost = distance
                    minref_text = ref_text
                    minref_speaker = ref_speaker
        
        if return_targets:
            targ_text.append(test_text)
            targ_speaker.append(test_speaker)
        pred_text.append(minref_text)
        pred_speaker.append(minref_speaker)
        unknow_filename.append(test_filename)

        if verbose == True and (i < 5 or (i % 50 == 0)):
            print(f'{i:3}/{len(test_dataloader)}: {pred_text[i]}')
    
    if return_targets : return np.array(pred_text), np.array(pred_speaker), np.array(targ_text), np.array(targ_speaker), unknow_filename
    return np.array(pred_text), np.array(pred_speaker), unknow_filename


def wer(pred, targets):

    v_err = np.zeros(len(pred))
    v_err[pred != targets] = 1
    err = np.sum(v_err)

    wer = 100 * err / len(pred)

    return wer


def main(commands10x10_list, commands10x100_list, free10x4x4_list, test_wavs_list, output_path):

    commands10x10 = DTW_Dataset(commands10x10_list, data_root='', names=None, type_='train')
    commands10x100 = DTW_Dataset(commands10x100_list, data_root='', names=None, type_='train')
    free10x4x4 = DTW_Dataset(free10x4x4_list, data_root='', names=None, type_='train')
    test_wavs = DTW_Dataset(test_wavs_list, data_root='', names=None, type_='test')

    mfsc_funct = lambda y, sfr : mfsc(y, sfr, window_size=0.025, window_stride=0.010, window='hamming', normalize=False, log=True, n_mels=20, preemCoef=0, melfloor=1.0, n_fft=512)
    # TODO mfcc with deltas as def funct as mfsc2mfcc_funct
    mfsc2mfcc_funct = lambda S : mfsc2mfcc(S, n_mfcc=12, dct_type=2, norm='ortho', lifter=22, cms=True, cmvn=True)

    commands10x10_mfcc = DTW_MFCC_Dataset(commands10x10, mfsc_funct=mfsc_funct, mfsc2mfcc_funct=mfsc2mfcc_funct, return_wav=True)
    commands10x100_mfcc = DTW_MFCC_Dataset(commands10x100, mfsc_funct=mfsc_funct, mfsc2mfcc_funct=mfsc2mfcc_funct, return_wav=True)
    free10x4x4_mfcc = DTW_MFCC_Dataset(free10x4x4, mfsc_funct=mfsc_funct, mfsc2mfcc_funct=mfsc2mfcc_funct, return_wav=True)
    test_wavs_mfcc = DTW_MFCC_Dataset(test_wavs, mfsc_funct=mfsc_funct, mfsc2mfcc_funct=mfsc2mfcc_funct, return_wav=True)

    # Free Spoken Digit Dataset
    free10x4x4_loader = DataLoader(free10x4x4_mfcc, collate_fn=build_dtw_collate(), batch_size=1)
    pred_text, pred_speaker, targ_text, targ_speaker, _ = predict(free10x4x4_loader, ref_dataloader=None, same_spk=True, return_targets=True, verbose=False)
    print(f'Text WER including reference recordings from the same speaker: {wer(pred_text, targ_text):.1f}%')
    print(f'Speaker WER including reference recordings from the same speaker: {wer(pred_speaker, targ_speaker):.1f}%')

    # Google Speech Commands Dataset (small digit subset)
    commands10x100_loader = DataLoader(commands10x100_mfcc, collate_fn=build_dtw_collate(), batch_size=1)
    pred_text, pred_speaker, targ_text, targ_speaker, _ = predict(commands10x100_loader, ref_dataloader=None, same_spk=False, return_targets=True, verbose=False)
    print(f'Text WER using only reference recordings from other speakers: {wer(pred_text, targ_text):.1f}%')
    print(f'Speaker WER using only reference recordings from other speakers: {wer(pred_speaker, targ_speaker):.1f}%')

    test_wavs_loader = DataLoader(test_wavs_mfcc, collate_fn=build_dtw_collate(), batch_size=1)
    test_ref_loader = DataLoader(ConcatDataset([commands10x10_mfcc, commands10x100_mfcc]), collate_fn=build_dtw_collate(), batch_size=1)
    pred_text, pred_speaker, unknow_filename = predict(test_wavs_loader, ref_dataloader=test_ref_loader, same_spk=True, return_targets=False, verbose=True)

    with open(output_path, 'w') as f:
        print('filename,command', file=f)

        for command, entry in zip(pred_text, unknow_filename):
            filename = entry.split('/')[-1].split('.')[0]

            print(f'{filename},{command}', file=f)


if __name__ == '__main__':

    commands10x10_list = './data_lists/commands10x10.csv'
    commands10x100_list = './data_lists/commands10x100.csv'
    free10x4x4_list = './data_lists/free10x4x4.csv'
    test_wavs_list = './data_lists/test_wavs.csv'

    output_name = datetime.now().strftime(f"%Y%m%d%H%M%S_submission")
    output_path = f'./dtw3/{output_name}.csv'
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    main(commands10x10_list, commands10x100_list, free10x4x4_list, test_wavs_list, output_path)
