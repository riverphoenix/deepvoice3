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


def plot_losses_world(config,Kmel_out,Ky1,pitch,y4a,harmonic,y4b,aperiodic,y4c,gs):
    plt.figure(figsize=(10, 10))

    plt.subplot(4, 2, 1)
    librosa.display.specshow(Kmel_out[0,:,:], y_axis='log')
    plt.title('Predicted mel')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 2)
    librosa.display.specshow(Ky1[0,:,:], y_axis='log')
    plt.title('Original mel')
    plt.colorbar()
    plt.tight_layout()

    ind = np.arange(len(pitch[0,:]))
    width = 0.35

    ax = plt.subplot(4, 2, 3)
    ax.bar(ind, pitch[0,:], width, color='r')
    plt.title('Predicted Pitch')
    plt.tight_layout()
  
    ax = plt.subplot(4, 2, 4)
    ax.bar(ind, y4a[0,:], width, color='r')
    plt.title('Original Pitch')
    plt.tight_layout()

    plt.subplot(4, 2, 5)
    librosa.display.specshow(harmonic[0,:,:].T,y_axis='log')
    plt.title('Predicted harmonic')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 6)
    librosa.display.specshow(y4b[0,:,:].T,y_axis='log')
    plt.title('Original harmonic')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 7)
    librosa.display.specshow(aperiodic[0,:,:].T)
    plt.title('Predicted Aperiodic')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 2, 8)
    librosa.display.specshow(y4c[0,:,:].T)
    plt.title('Original Aperiodic')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig('{}/losses_world_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_wavs(config,wavs,gs):
    if len(wavs)!=0:
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