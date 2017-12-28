# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
from __future__ import print_function

import numpy as np
import librosa
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import step, show
import librosa.display

from scipy.signal import freqz
from scipy.signal import butter, lfilter
from hyperparams import Hyperparams as hp
from scipy.io.wavfile import write

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-normalize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    #mag = (np.clip(mag, 0, 1) * (hp.max_db_mag - hp.ref_db_mag)) + hp.ref_db_mag


    # to amplitude
    mag = librosa.db_to_amplitude(mag)
    # print(np.max(mag), np.min(mag), mag.shape)
    # (1025, 812, 16)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    wav = butter_bandpass_filter(wav, hp.lowcut, hp.highcut, hp.sr, order=6)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_loss(config,mag,y,gs):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(mag[0,:,:].T)
    plt.title('Predicted mag')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(y[0,:,:].T)
    plt.title('Original mag')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig('{}/losses_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_wavs(config,wav1,wav2,gs):
    plt.figure(figsize=(10, 10))
    plt.subplot(2,1,1)
    librosa.display.waveplot(wav1, sr=hp.sr)
    plt.subplot(2,1,2)
    librosa.display.waveplot(wav2, sr=hp.sr)

    plt.savefig('{}/wavs_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def get_wavs(config,mag,y,gs):
    mag = np.squeeze(mag[0])
    wav_mag = spectrogram2wav(mag)
    y = np.squeeze(y[0])
    wav_y = spectrogram2wav(y)
    return wav_mag, wav_y

def save_wavs(config,wav_mag,wav_y,gs):
    write(config.log_dir + "/{}_predict.wav".format(gs), hp.sr, wav_mag)
    write(config.log_dir + "/{}_original.wav".format(gs), hp.sr, wav_y)