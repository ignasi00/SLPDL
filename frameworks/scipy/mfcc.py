
from functools import lru_cache
import numpy as np
import scipy.fftpack

from .filters import apply_preemphasis
from .utils import normalized
from .windows import get_windows


def freq_to_mel(freq) : return 2595.0 * np.log10(1.0 + freq / 700.0)
def mel_to_freq(mels) : return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)

@lru_cache(maxsize=10)
def get_filterbank(numfilters, filterLen, lowFreq, highFreq, samplingFreq):

    minwarpfreq = freq_to_mel(lowFreq)
    maxwarpfreq = freq_to_mel(highFreq)
    
    dwarp = (maxwarpfreq - minwarpfreq) / (numfilters + 1)
    
    f = mel_to_freq(np.arange(numfilters + 2) * dwarp + minwarpfreq) * (filterLen - 1) * 2.0 / samplingFreq

    i = np.arange(filterLen)[None, :]
    f = f[:, None]
    
    hislope = (i - f[:numfilters]) / (f[1:numfilters+1] - f[:numfilters])
    loslope = (f[2:numfilters+2] - i) / (f[2:numfilters+2] - f[1:numfilters+1])
    
    H = np.maximum(0, np.minimum(hislope, loslope))
    
    return H

def mfsc(y, sfr, window_size=0.025, window_stride=0.010, window='hamming', normalize=False, log=True, n_mels=20, preemCoef=0, melfloor=1.0, n_fft=512):
    # window length in samples
    win_length = int(sfr * window_size)
    
    # window shift in samples
    hop_length = int(sfr * window_stride)
    
    # frequency analysis limits
    lowfreq = 0
    highfreq = sfr/2
    
    # get window
    window = get_windows(win_length, type_=window)
    try:
        padded_window = np.pad(window, (0, n_fft - win_length), mode='constant')[:, None]
    except Exception as e:
        if n_fft - win_length < 0:
            raise Exception(f"Not enough n_fft points (currently: {n_fft})\nAt least it should be {win_length}")
        raise e
    
    # preemphasis
    y = apply_preemphasis(y.copy(), preemCoef)

    # scale wave signal; IGNASI: ¿?
    y *= 32768
    
    # get overlaped frames using numpy stride_tricks
    num_frames = 1 + (len(y) - win_length) // hop_length
    pad_after = num_frames * hop_length + (n_fft - hop_length) - len(y)
    if pad_after > 0:
        y = np.pad(y, (0, pad_after), mode='constant')
    frames = np.lib.stride_tricks.as_strided(y, shape=(n_fft, num_frames), strides=(y.itemsize, hop_length * y.itemsize), writeable=False)
    windowed_frames = padded_window * frames
    
    # compute the modulus of the DFT of each frame
    D = np.abs(np.fft.rfft(windowed_frames, axis=0))

    # apply mel filterbank
    filterbank = get_filterbank(n_mels, n_fft/2 + 1, lowfreq, highfreq, sfr)
    mf = np.dot(filterbank, D)
    
    if log:
        mf = np.log(np.maximum(melfloor, mf))
    if normalize:
        mf = normalized(mf)

    return mf

def mfsc2mfcc(S, n_mfcc=12, dct_type=2, norm='ortho', lifter=22, cms=True, cmvn=True):
    # Discrete Cosine Transform
    M = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc]

    # Ceptral mean subtraction (CMS) 
    if cms or cmvn:
        M -= M.mean(axis=1, keepdims=True)

    # Ceptral mean and variance normalization (CMVN)
    if cmvn:
        M /= M.std(axis=1, keepdims=True)
    
    # Liftering
    elif lifter > 0:
        lifter_window = 1 + (lifter / 2) * np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)[:, np.newaxis]
        M *= lifter_window

    return M
