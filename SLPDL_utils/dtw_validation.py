
from frameworks.scipy.dtw import dtw


# Word Error Rate (Accuracy)
def wer(test_dataloader, ref_dataloader=None, same_spk=False):
    # the dataloaders have to be "DTW_MFCC_Dataset with return_wav=True" and "batch_size=1"; TODO: batch_size > 1 (remember drop_last=True)

    assert test_dataloader.batch_size == 1

    if ref_dataloader is None:
        ref_dataloader = test_dataloader
        
    err = 0
    for i, test_mfcc, test_wav, test_sfr, test_text, test_speaker in enumerate(test_dataloader):
        mincost = np.inf
        minref_text = None
        for j, ref_mfcc, ref_wav, ref_sfr, ref_text, ref_speaker in enumerate(ref_dataloader):

            if not same_spk and test_speaker == ref_speaker:
                # Do not compare with refrence recordings of the same speaker
                continue

            if test_wav != ref_wav:
                distance = dtw(test_mfcc, ref_mfcc)

                if distance < mincost:
                    mincost = distance
                    minref_text = ref_text
                    
        if test_text != minref_text:
            err += 1

    wer = 100 * err / len(test_dataloader)
    # wer = 100 * err / (len(test_dataloader) * test_dataloader.batch_size)

    return wer
