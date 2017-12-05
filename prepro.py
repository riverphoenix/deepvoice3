# -*- coding: utf-8 -*-
# #/usr/bin/python2

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

import numpy as np
import librosa

from hyperparams import Hyperparams as hp
import glob
import os
import tqdm

import magphase.magphase as mp
import pyworld as pw
import multiprocessing

def get_spectrograms(sound_file):
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y_pre = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y_pre,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = librosa.amplitude_to_db(mel)

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)

    return mel, y

def prep_all_files(files):

    for file in tqdm.tqdm(files):
        fname = os.path.basename(file)
        
        mel, input_x = get_spectrograms(file)
        np.save(os.path.join(mel_folder, fname.replace(".wav", ".npy")), mel)

        pitch, harmonic, aperiodic = pw.wav2world(np.float64(input_x), hp.sr)
        pitch = pitch.astype(np.float32)
        harmonic = harmonic.astype(np.float32)
        aperiodic = aperiodic.astype(np.float32)

        np.save(os.path.join(pitch_folder, fname.replace(".wav", ".npy")), pitch)
        np.save(os.path.join(harmonic_folder, fname.replace(".wav", ".npy")), harmonic)
        np.save(os.path.join(aperiodic_folder, fname.replace(".wav", ".npy")), aperiodic)

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

if __name__ == "__main__":
    wav_folder = os.path.join(hp.data, 'wavs')
    mel_folder = os.path.join(hp.data, 'mels')

    pitch_folder = os.path.join(hp.data, 'pitches')
    harmonic_folder = os.path.join(hp.data, 'harmonics')
    aperiodic_folder = os.path.join(hp.data, 'aperiodics')    

    for folder in (mel_folder, pitch_folder, harmonic_folder, aperiodic_folder):
        if not os.path.exists(folder): os.mkdir(folder)

    files = glob.glob(os.path.join(wav_folder, "*"))
    if hp.prepro_gpu > 1:
        files = split_list(files, wanted_parts=hp.prepro_gpu)
        for i in range(hp.prepro_gpu):
            p = multiprocessing.Process(target=prep_all_files, args=(files[i],))
            p.start()
    else:
        prep_all_files(files)