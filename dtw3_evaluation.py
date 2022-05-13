
# TODO: remake the list without random shuffling

import torch
from torch.utils.data import ConcatDataset

import warnings
from numba.core.errors import NumbaWarning
warnings.simplefilter("ignore", category=NumbaWarning)

from SLPDL_utils.dtw_dataset import DTW_Dataset, DTW_MFCC_Dataset, build_dtw_collate


def predict(test_dataloader, ref_dataloader=None, same_spk=False, return_targets=False, verbose=False):
    # the dataloaders have to be "DTW_MFCC_Dataset with return_wav=True" and "batch_size=1"; TODO: batch_size > 1 

    assert test_dataloader.batch_size == 1
    if ref_dataloader is None : ref_dataloader = test_dataloader
        
    pred_text = []
    pred_speaker = []
    targ_text = []
    targ_speaker = []
    for i, test_data in enumerate(test_dataloader):
        if same_spk == False or test_speaker == True:
            test_mfcc, test_wav, test_sfr, test_filename, test_text, test_speaker = test_data
        else:
            test_mfcc, test_wav, test_sfr, test_filename = test_data

        if return_targets:
            targ_text.append(test_text)
            targ_speaker.append(test_speaker)

        mincost = np.inf
        minref_text = None
        minref_speaker = None
        for j, ref_mfcc, ref_wav, ref_sfr, ref_filename, ref_text, ref_speaker in enumerate(ref_dataloader):

            if not same_spk and test_speaker == ref_speaker:
                # Do not compare with refrence recordings of the same speaker
                continue

            if test_wav != ref_wav:
                distance = dtw(test_mfcc, ref_mfcc)

                if distance < mincost:
                    mincost = distance
                    minref_text = ref_text
                    minref_speaker = ref_speaker
        
        pred_text.append(minref_text)
        pred_speaker.append(minref_speaker)

        if verbose == True and i < 10:
            print(f'{i:3}/{len(test_dataloader)}: {pred_text[i]}')
    
    if return_targets : return pred_text, pred_speaker, targ_text, targ_speaker
    return pred_text, pred_speaker


def wer(pred, targets):
    v_err = np.zeros_like(pred)
    v_err[pred != targets] = 1
    err = np.sum(v_err)

    wer = 100 * err / len(pred)

    return wer


def main(commands10x10_list, commands10x100_list, free10x4x4_list, test_wavs_list):

    commands10x10 = DTW_Dataset(commands10x10_list, data_root='', names=None, type_='train')
    commands10x100 = DTW_Dataset(commands10x100_list, data_root='', names=None, type_='train')
    free10x4x4 = DTW_Dataset(free10x4x4_list, data_root='', names=None, type_='train')
    test_wavs = DTW_Dataset(test_wavs_list, data_root='', names=None, type_='test')

    commands10x10_mfcc = DTW_MFCC_Dataset(commands10x10, mfsc_funct=None, return_wav=False)
    commands10x100_mfcc = DTW_MFCC_Dataset(commands10x100, mfsc_funct=None, return_wav=False)
    free10x4x4_mfcc = DTW_MFCC_Dataset(free10x4x4, mfsc_funct=None, return_wav=False)
    test_wavs_mfcc = DTW_MFCC_Dataset(test_wavs, mfsc_funct=None, return_wav=False)

    # Free Spoken Digit Dataset
    free10x4x4_loader = DataLoader(free10x4x4_mfcc, collate_fn=build_dtw_collate(), batch_size=1)
    pred_text, pred_speaker, targ_text, targ_speaker = predict(free10x4x4_loader, ref_dataloader=None, same_spk=True, return_targets=True, verbose=False)
    print(f'WER including reference recordings from the same speaker: {wer(pred_text, targ_text):.1f}%')

    # Google Speech Commands Dataset (small digit subset)
    commands10x100_loader = DataLoader(commands10x100_mfcc, collate_fn=build_dtw_collate(), batch_size=1)
    pred_text, pred_speaker, targ_text, targ_speaker = predict(commands10x100_loader, ref_dataloader=None, same_spk=True, return_targets=True, verbose=False)
    print(f'WER using only reference recordings from other speakers: {wer(pred_text, targ_text):.1f}%')

    test_wavs_loader = DataLoader(test_wavs_mfcc, collate_fn=build_dtw_collate(), batch_size=1)
    test_ref_loader = DataLoader(ConcatDataset([commands10x10, commands10x100]), collate_fn=build_dtw_collate(), batch_size=1)
    pred_text, pred_speaker = predict(test_wavs_loader, ref_dataloader=test_ref_loader, same_spk=False, return_targets=False, verbose=True)

    with open('submission.csv', 'w') as f:
        print('filename,command', file=f)

        for i, command in enumerate(pred_text):
            entry = test_wavs_loader[i][2]

            filename = entry['wav'].split('/')[-1].split('.')[0]

            print(f'{filename},{command}', file=f)


if __name__ == '__main__':

    commands10x10_list = './data_lists/commands10x10.csv'
    commands10x100_list = './data_lists/commands10x100.csv'
    free10x4x4_list = './data_lists/free10x4x4.csv'
    test_wavs_list = './data_lists/test_wavs.csv'

    main(commands10x10_list, commands10x100_list, free10x4x4_list, test_wavs_list)
