# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
from __future__ import print_function
from numpy import inf

import numpy as np
import librosa
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperparams import Hyperparams as hp


def spectrogram2wav(spectrogram):
    '''Convert spectrogram into a waveform using Griffin-lim's raw.
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        X_t = np.nan_to_num(X_t)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignments, gs, elapsed_time):
    """
    Plots the alignment
    alignment: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    elasped_time: seconds.
    """
    hours = elapsed_time // 3600
    minutes = (elapsed_time - 3600 * hours) // 60

    fig, axes = plt.subplots(nrows=len(alignments), ncols=1, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(alignments[i])
        ax.axis('off')
        ax.set_title("Layer {}".format(i))

    fig.subplots_adjust(right=0.8, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('{} Steps After {} hours {} minutes'.format(gs, hours, minutes))
    plt.savefig('{}/alignment_{}.png'.format(hp.logdir, gs), format='png')
def plot_alignment2(alignments, gs, elapsed_time):
    """
    Plots the alignment
    alignment: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    elasped_time: seconds.
    """
    hours = elapsed_time // 3600
    minutes = (elapsed_time - 3600 * hours) // 60

    fig, axes = plt.subplots(nrows=len(alignments), ncols=1, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        im = ax.plot(alignments[i])
        ax.axis('off')
        ax.set_title("Layer {}".format(i))

    fig.subplots_adjust(right=0.8, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('{} Steps After {} hours {} minutes'.format(gs, hours, minutes))
    plt.savefig('{}/alignment_{}.png'.format(hp.logdir, gs), format='png')
