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
import multiprocessing


def get_spectrograms(sound_file):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # Sequence length
    done = np.ones_like(mel[0, :]).astype(np.int32)

    # to decibel
    mel = librosa.amplitude_to_db(mel)
    mag = librosa.amplitude_to_db(mag)

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, done, mag

def prep_all_files(files):

    for file in tqdm.tqdm(files):
        fname = os.path.basename(file)
        
        mel, dones, mag = get_spectrograms(file)
        np.save(os.path.join(mel_folder, fname.replace(".wav", ".npy")), mel)
        np.save(os.path.join(dones_folder, fname.replace(".wav", ".npy")), dones)
        np.save(os.path.join(mag_folder, fname.replace(".wav", ".npy")), mag)
        
        magmel, realmel, imagmel, freq, _, _, _ = mp.analysis_compressed(file,fft_len=hp.n_fft,
            nbins_mel=hp.n_mels, nbins_phase=hp.nbins_phase)
        magmel = magmel.astype(np.float32)
        realmel = realmel.astype(np.float32)
        imagmel = imagmel.astype(np.float32)
        freq = freq.astype(np.float32)

        np.save(os.path.join(magmel_folder, fname.replace(".wav", ".npy")), magmel)
        np.save(os.path.join(realmel_folder, fname.replace(".wav", ".npy")), realmel)
        np.save(os.path.join(imagmel_folder, fname.replace(".wav", ".npy")), imagmel)
        np.save(os.path.join(freq_folder, fname.replace(".wav", ".npy")), freq)

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

if __name__ == "__main__":
    wav_folder = os.path.join(hp.data, 'wavs')
    mel_folder = os.path.join(hp.data, 'mels')
    dones_folder = os.path.join(hp.data, 'dones')
    mag_folder = os.path.join(hp.data, 'mags')

    magmel_folder = os.path.join(hp.data, 'magmels')
    realmel_folder = os.path.join(hp.data, 'realmels')
    imagmel_folder = os.path.join(hp.data, 'imagmels')    
    freq_folder = os.path.join(hp.data, 'freqs')

    for folder in (mel_folder, dones_folder, mag_folder,magmel_folder,realmel_folder,imagmel_folder,freq_folder):
        if not os.path.exists(folder): os.mkdir(folder)

    files = glob.glob(os.path.join(wav_folder, "*"))
    if hp.prepro_gpu > 1:
        files = split_list(files, wanted_parts=hp.prepro_gpu)
        for i in range(hp.prepro_gpu):
            p = multiprocessing.Process(target=prep_all_files, args=(files[i],))
            p.start()
    else:
        prep_all_files(files)