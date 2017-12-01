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

from hyperparams import Hyperparams as hp


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = librosa.db_to_amplitude(mag)
    # print(np.max(mag), np.min(mag), mag.shape)
    # (1025, 812, 16)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

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

def plot_alignment(config,alignments, gs):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, axes = plt.subplots(nrows=len(alignments), ncols=1, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        # i=0
        im = ax.imshow(alignments[i])
        ax.axis('off')
        ax.set_title("Layer {}".format(i))

    fig.subplots_adjust(right=0.8, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_losses(config,Kmel_out,Ky1,KDone,Ky2,KMag,Kz,gs):
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 2, 1)
    librosa.display.specshow(Kmel_out[0,:,:])
    plt.title('Predicted mel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 2, 2)
    librosa.display.specshow(Ky1[0,:,:])
    plt.title('Original mel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 2, 3)
    librosa.display.specshow(KMag[0,:,:], y_axis='log')
    #plt.colorbar(format='%+2.0f dB')
    plt.title('Predicted mag')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 2, 4)
    librosa.display.specshow(Kz[0,:,:], y_axis='log')
    #plt.colorbar(format='%+2.0f dB')
    plt.title('Original mag')
    plt.colorbar()
    plt.tight_layout()

    KDone = KDone[0,:,:]
    Kd = []
    for i in range(KDone.shape[0]):
        if KDone[i,0] > KDone[i,1]:
            Kd.append(0)
        else:
            Kd.append(1)

    ind = np.arange(len(Kd))
    width = 0.35

    ax = plt.subplot(3, 2, 5)
    ax.bar(ind, Kd, width, color='r')
    plt.title('Predicted Dones')
    plt.tight_layout()
  
    ax = plt.subplot(3, 2, 6)
    ax.bar(ind, Ky2[0,:], width, color='r')
    plt.title('Original Dones')
    plt.tight_layout()

    plt.savefig('{}/losses_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_losses2(config,Kmel_out,Ky1,KDone,Ky2,gs):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    librosa.display.specshow(Kmel_out[0,:,:])
    plt.title('Predicted mel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    librosa.display.specshow(Ky1[0,:,:])
    plt.title('Original mel')
    plt.colorbar()
    plt.tight_layout()

    KDone = KDone[0,:,:]
    Kd = []
    for i in range(KDone.shape[0]):
        if KDone[i,0] > KDone[i,1]:
            Kd.append(0)
        else:
            Kd.append(1)

    ind = np.arange(len(Kd))
    width = 0.35

    ax = plt.subplot(2, 2, 3)
    ax.bar(ind, Kd, width, color='r')
    plt.title('Predicted Dones')
    plt.tight_layout()
  
    ax = plt.subplot(2, 2, 4)
    ax.bar(ind, Ky2[0,:], width, color='r')
    plt.title('Original Dones')
    plt.tight_layout()

    plt.savefig('{}/losses_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_losses_magphase(config,magmel,y3a,realmel,y3b,imagemel,y3c,freq,y3d,gs):
    plt.figure(figsize=(10, 10))

    plt.subplot(4, 2, 1)
    librosa.display.specshow(magmel[0,:,:],y_axis='log')
    plt.title('Predicted magmel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 2)
    librosa.display.specshow(y3a[0,:,:],y_axis='log')
    plt.title('Original magmel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 3)
    librosa.display.specshow(realmel[0,:,:],y_axis='log')
    plt.title('Predicted realmel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 4)
    librosa.display.specshow(y3b[0,:,:],y_axis='log')
    plt.title('Original realmel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 5)
    librosa.display.specshow(imagemel[0,:,:],y_axis='log')
    plt.title('Predicted imagemel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 6)
    librosa.display.specshow(y3c[0,:,:],y_axis='log')
    plt.title('Original imagmel')
    plt.colorbar()
    plt.tight_layout()

    ind = np.arange(len(freq[0,:]))
    width = 0.35

    ax = plt.subplot(4, 2, 7)
    ax.bar(ind, freq[0,:], width, color='r')
    plt.title('Predicted Freq')
    plt.tight_layout()
  
    ax = plt.subplot(4, 2, 8)
    ax.bar(ind, y3d[0,:], width, color='r')
    plt.title('Original Freq')
    plt.tight_layout()

    plt.savefig('{}/losses_magphase_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_losses_world(config,pitch,y4a,harmonic,y4b,aperiodic,y4c,gs):
    plt.figure(figsize=(10, 10))

    ind = np.arange(len(pitch[0,:]))
    width = 0.35

    ax = plt.subplot(3, 2, 1)
    ax.bar(ind, pitch[0,:], width, color='r')
    plt.title('Predicted Pitch')
    plt.tight_layout()
  
    ax = plt.subplot(3, 2, 2)
    ax.bar(ind, y4a[0,:], width, color='r')
    plt.title('Original Pitch')
    plt.tight_layout()

    plt.subplot(3, 2, 3)
    librosa.display.specshow(harmonic[0,:,:],y_axis='log')
    plt.title('Predicted harmonic')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 2, 4)
    librosa.display.specshow(y4b[0,:,:],y_axis='log')
    plt.title('Original harmonic')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 2, 5)
    librosa.display.specshow(aperiodic[0,:,:])
    plt.title('Predicted Aperiodic')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 2, 6)
    librosa.display.specshow(y4c[0,:,:])
    plt.title('Original Aperiodic')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig('{}/losses_world_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_wavs(config,wavs,gs):
    plt.figure(figsize=(10, 10))
    for i in range(len(wavs)):
        wav = wavs[i]
        txt = str(wav[2])+':'+str(wav[0])
        wv = wav[1]

        plt.subplot(len(wavs),1, i+1)
        librosa.display.waveplot(wv, sr=hp.sr)
        plt.title(txt)
    plt.savefig('{}/wavs_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')