# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
from num2words import num2words
from random import randint
import pandas as pd

cmu = pd.read_csv('cmudict.dict.txt',header=None,names=['name'])
cmu['word'], cmu['phone'] = cmu['name'].str.split(' ', 1).str
cmu['word'] = cmu['word'].str.upper()
cmu.drop(['name'],axis=1,inplace=True)
cmu = list(cmu.set_index('word').to_dict().values()).pop()

def text_normalize(sent):
    '''Minimum text preprocessing'''
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    normalized = []
    for word in sent.split():
        word = _strip_accents(word.lower())
        srch = re.match("\d[\d,.]*$", word)
        if srch:
            word = num2words(float(word.replace(",", "")))
        word = re.sub(u"[-—-]", " ", word)
        word = re.sub("[^ a-z'.?]", "", word)
        normalized.append(word)
    normalized = " ".join(normalized)
    normalized = re.sub("[ ]{2,}", " ", normalized)
    normalized = normalized.strip()
    return normalized

def text_normalize_cmu(sent):
    '''Remove accents and upper strings.'''
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    normalized = []
    for word in sent.split():
        word = _strip_accents(word.lower())
        srch = re.match("\d[\d,.]*$", word)
        if srch:
            word = num2words(float(word.replace(",", "")))
        word = re.sub(u"[-—-]", " ", word)
        word = re.sub("[^ a-z'.?]", "", word)
        normalized.append(word)
    normalized = " ".join(normalized)
    normalized = re.sub("[ ]{2,}", " ", normalized)
    normalized = normalized.strip()

    normalized = re.sub("[^ A-Z,;.]", "", _strip_accents(sent).upper())
    if normalized[-1] in [".",",","?",";"]:
        normalized = normalized[0:-1]
    normalized = re.sub('\'',' ',normalized)
    normalized = re.sub(' ','@',normalized)
    normalized = re.sub(',','@@',normalized)
    normalized = re.sub(';','@@@',normalized)
    normalized = re.sub('\.','@@@@',normalized)
    normalized = normalized.strip()
    return normalized

def break_to_phonemes(strin):
    strin = re.sub('([A-Z])@','\\1 @',strin)
    strin = re.sub('([A-Z])\*','\\1 *',strin)
    strin = re.sub('@([A-Z])','@ \\1',strin)
    strin = re.sub("\\s+", " ",strin)
    strin = re.split('\s',strin)
    strout = ""
    for word_in in strin:
        word_in = word_in.upper()
        wpd = wwd = ""
        if "@" in word_in:
            wpd = word_in
        else:
            if word_in in cmu:
                wwd = cmu[word_in].split(" ")
                for kl in range(0,len(wwd)):
                    if len(wwd[kl])==3:
                        wwd[kl] = wwd[kl][0:2]
            else:
                wwd = list(word_in)
            for kl in range(0,len(wwd)):
                if kl!=len(wwd)-1:
                    wpd = wpd+wwd[kl]+" "
                else:
                    wpd = wpd+wwd[kl]
        strout = strout + wpd
    return strout

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_vocab_cmu():
    valid_symbols = ['#','@','A','AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'C','CH', 'D', 'DH', 'E','EH', 'ER', 'EY',
    'F', 'G', 'H','HH', 'I','IH', 'IY', 'J','JH', 'K', 'L', 'M', 'N', 'NG', 'OW','O', 'OY', 'P', 'Q','R', 'S', 'SH',
    'T', 'TH', 'U','UH', 'UW','V', 'W', 'X','Y', 'Z', 'ZH','*',"'"]
    _valid_symbol_set = set(valid_symbols)
    
    char2idx = {char: idx for idx, char in enumerate(_valid_symbol_set)}
    idx2char = {idx: char for idx, char in enumerate(_valid_symbol_set)}
    
    return char2idx, idx2char

def str_to_ph(strin):
    strin = re.sub('([A-Z])@','\\1 @',strin)
    strin = re.sub('([A-Z])\*','\\1 *',strin)
    strin = re.sub('@([A-Z])','@ \\1',strin)
    strin = re.sub('@',' @',strin)
    strin = re.sub("\\s+", " ",strin)
    strin = re.sub("@\*","*",strin)
    strin = re.split('\s',strin)
    return strin

def invert_text(txt):
    if not hp.run_cmu: 
        char2idx, idx2char = load_vocab()
        pstring = [idx2char[char] for char in txt]
        pstring = ''.join(pstring)
        pstring = pstring.replace("E", "")
        pstring = pstring.replace("P", "")
    else:
        char2idx, idx2char = load_vocab_cmu() 
        pstring = [idx2char[char] for char in txt]
        pstring = ''.join(pstring)
        pstring = pstring.replace("@", " ")
        pstring = pstring.replace("#", "")
        pstring = pstring.replace("*", "")

    return pstring


def load_test_data():
    # Load vocabulary
    if not hp.run_cmu: 
        char2idx, idx2char = load_vocab()
    else:
        char2idx, idx2char = load_vocab_cmu() 

    # Parse
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        if not hp.run_cmu: 
            sent = text_normalize(line).strip() + "E" # text normalization, E: EOS
        else:
            sent = text_normalize_cmu(line) + "*" # text normalization, *: EOS
            sent = break_to_phonemes(sent)
            sent = str_to_ph(sent)
        if len(sent) <= hp.T_x:
            if not hp.run_cmu: 
                sent += "P"*(hp.T_x-len(sent))
            else:
                sent.extend(['#'] * (hp.T_x-len(sent)))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts

def load_data(config,training=True):
    # Load vocabulary
    if not hp.run_cmu: 
        char2idx, idx2char = load_vocab()
    else:
        char2idx, idx2char = load_vocab_cmu()    

    # Parse
    texts, _texts_test, mels, dones, mags, magmels, realmels, imagmels, freqs, pitches, harmonics, aperiodics = [], [], [], [], [], [], [], [], [], [], [], []
    num_samples = 1
    metadata = os.path.join(config.data_paths, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        fname, _, sent = line.strip().split("|")
        if not hp.run_cmu: 
            sent = text_normalize(sent) + "E" # text normalization, E: EOS
        else:
            sent = text_normalize_cmu(sent) + "*" # text normalization, E: EOS
            sent = break_to_phonemes(sent)
            sent = str_to_ph(sent)
        if len(sent) <= hp.T_x:
            if not hp.run_cmu: 
                sent += "P"*(hp.T_x-len(sent)) #this was added
            else:
                sent.extend(['#'] * (hp.T_x-len(sent)))
            pstring = [char2idx[char] for char in sent]  
            texts.append(np.array(pstring, np.int32).tostring())
            _texts_test.append(np.array(pstring,np.int32).tostring())
            mels.append(os.path.join(config.data_paths, "mels", fname + ".npy"))
            dones.append(os.path.join(config.data_paths, "dones", fname + ".npy"))
            if hp.predict_griffin:
                mags.append(os.path.join(config.data_paths, "mags", fname + ".npy"))
            if  hp.predict_melograph:
                magmels.append(os.path.join(config.data_paths, "magmels", fname + ".npy"))
                realmels.append(os.path.join(config.data_paths, "realmels", fname + ".npy"))
                imagmels.append(os.path.join(config.data_paths, "imagmels", fname + ".npy"))
                freqs.append(os.path.join(config.data_paths, "freqs", fname + ".npy"))
            if hp.predict_world:
                pitches.append(os.path.join(config.data_paths, "pitches", fname + ".npy"))
                harmonics.append(os.path.join(config.data_paths, "harmonics", fname + ".npy"))
                aperiodics.append(os.path.join(config.data_paths, "aperiodics", fname + ".npy"))

    return texts, _texts_test, mels, dones, mags, magmels, realmels, imagmels, freqs, pitches, harmonics, aperiodics

def get_batch(config):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _texts_tests, _mels, _dones, _mags, _magmels, _realmels, _imagmels, _freqs, _pitches, _harmonics, _aperiodics = load_data(config)

        # Calc total batch count
        num_batch = len(_texts) // hp.batch_size
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        texts_tests = tf.convert_to_tensor(_texts_tests)
        mels = tf.convert_to_tensor(_mels)
        dones = tf.convert_to_tensor(_dones)
        if hp.predict_griffin:
            mags = tf.convert_to_tensor(_mags)
        if  hp.predict_melograph:
            magmels = tf.convert_to_tensor(_magmels)
            realmels = tf.convert_to_tensor(_realmels)
            imagmels = tf.convert_to_tensor(_imagmels)
            freqs = tf.convert_to_tensor(_freqs)
        if hp.predict_world:
            pitches = tf.convert_to_tensor(_pitches)
            harmonics = tf.convert_to_tensor(_harmonics)
            aperiodics = tf.convert_to_tensor(_aperiodics)

        # Create Queues
        if hp.predict_griffin:
            if hp.predict_melograph:
                if hp.predict_world:
                    text, texts_test, mel, done, mag, magmel, realmel, imagmel, freq, pitch, harmonic, aperiodic = tf.train.slice_input_producer([texts,texts_tests, mels, dones, mags, magmels, realmels, imagmels, freqs,pitches,harmonics,aperiodics], shuffle=False,capacity=hp.batch_size*32)
                else:
                    text, texts_test, mel, done, mag, magmel, realmel, imagmel, freq = tf.train.slice_input_producer([texts,texts_tests, mels, dones, mags, magmels, realmels, imagmels, freqs], shuffle=False,capacity=hp.batch_size*32)
            else:
                if hp.predict_world:
                    text, texts_test, mel, done, mag, pitch, harmonic, aperiodic = tf.train.slice_input_producer([texts,texts_tests, mels, dones, mags, pitches,harmonics,aperiodics], shuffle=False,capacity=hp.batch_size*32)
                else:
                    text, texts_test, mel, done, mag = tf.train.slice_input_producer([texts,texts_tests, mels, dones, mags], shuffle=False,capacity=hp.batch_size*32)
        else:
            if hp.predict_melograph:
                if hp.predict_world:
                    text, texts_test, mel, done, magmel, realmel, imagmel, freq, pitch, harmonic, aperiodic = tf.train.slice_input_producer([texts,texts_tests, mels, dones, magmels, realmels, imagmels, freqs,pitches,harmonics,aperiodics], shuffle=False,capacity=hp.batch_size*32)
                else:
                    text, texts_test, mel, done, magmel, realmel, imagmel, freq = tf.train.slice_input_producer([texts,texts_tests, mels, dones, magmels, realmels, imagmels, freqs], shuffle=False,capacity=hp.batch_size*32)
            else:
                if hp.predict_world:
                    text, texts_test, mel, done, pitch, harmonic, aperiodic = tf.train.slice_input_producer([texts,texts_tests, mels, dones, pitches,harmonics,aperiodics], shuffle=False,capacity=hp.batch_size*32)
                else:
                    text, texts_test, mel, done= tf.train.slice_input_producer([texts,texts_tests, mels, dones], shuffle=False,capacity=hp.batch_size*32)

        # Decoding
        text = tf.decode_raw(text, tf.int32) # (None,)
        texts_test = tf.decode_raw(texts_test, tf.int32) # (None,)
        mel = tf.py_func(lambda x:np.load(x), [mel], tf.float32) # (None, n_mels)
        done = tf.py_func(lambda x:np.load(x), [done], tf.int32) # (None,)
        if hp.predict_griffin:
            mag = tf.py_func(lambda x:np.load(x), [mag], tf.float32) # (None, 1+n_fft/2)
        if  hp.predict_melograph:
            magmel = tf.py_func(lambda x:np.load(x), [magmel], tf.float32)
            realmel = tf.py_func(lambda x:np.load(x), [realmel], tf.float32)
            imagmel = tf.py_func(lambda x:np.load(x), [imagmel], tf.float32)
            freq = tf.py_func(lambda x:np.load(x), [freq], tf.float32)
        if hp.predict_world:
            pitch = tf.py_func(lambda x:np.load(x), [pitch], tf.float32)
            harmonic = tf.py_func(lambda x:np.load(x), [harmonic], tf.float32)
            aperiodic = tf.py_func(lambda x:np.load(x), [aperiodic], tf.float32)

        # Padding
        text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x] # (Tx,)
        texts_test = tf.pad(texts_test, ((0, hp.T_x),))[:hp.T_x] # (Tx,)
        mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y] # (Ty, n_mels)
        done = tf.pad(done, ((0, hp.T_y),))[:hp.T_y] # (Ty,)
        if hp.predict_griffin:
            mag = tf.pad(mag, ((0, hp.T_y), (0, 0)))[:hp.T_y] # (Ty, 1+n_fft/2)
        if  hp.predict_melograph:
            magmel = tf.pad(magmel, ((0, hp.T_y2), (0, 0)))[:hp.T_y2]
            realmel = tf.pad(realmel, ((0, hp.T_y2), (0, 0)))[:hp.T_y2]
            imagmel = tf.pad(imagmel, ((0, hp.T_y2), (0, 0)))[:hp.T_y2]
            freq = tf.pad(freq, ((0, hp.T_y2),))[:hp.T_y2]
            freq = tf.nn.relu(freq)
        if hp.predict_world:
            pitch = tf.pad(pitch, ((0, hp.T_y3),))[:hp.T_y3]
            harmonic = tf.pad(harmonic, ((0, hp.T_y3), (0, 0)))[:hp.T_y3]
            aperiodic = tf.pad(aperiodic, ((0, hp.T_y3), (0, 0)))[:hp.T_y3]

        # Reduction
        mel = tf.reshape(mel, (hp.T_y//hp.r, -1)) # (Ty/r, n_mels*r)
        done = done[::hp.r] # (Ty/r,)

        if hp.predict_griffin:
            if hp.predict_melograph:
                if hp.predict_world:
                    texts, texts_tests, mels, dones, mags, magmels, realmels, imagmels, freqs, pitches, harmonics, aperiodics = tf.train.batch([text, texts_test, mel, done, mag, magmel, realmel, imagmel, freq, pitch, harmonic, aperiodic],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,), (hp.T_y, 1+hp.n_fft//2),
                                    (hp.T_y2,hp.n_mels),(hp.T_y2,hp.nbins_phase),(hp.T_y2,hp.nbins_phase),(hp.T_y2,),(hp.T_y3,),(hp.T_y3,hp.world_d),(hp.T_y3,hp.world_d)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, mags, magmels, realmels, imagmels, freqs, pitches, harmonics, aperiodics, num_batch
                else:
                    texts, texts_tests, mels, dones, mags, magmels, realmels, imagmels, freqs = tf.train.batch([text, texts_test, mel, done, mag, magmel, realmel, imagmel, freq],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,), (hp.T_y, 1+hp.n_fft//2),
                                    (hp.T_y2,hp.n_mels),(hp.T_y2,hp.nbins_phase),(hp.T_y2,hp.nbins_phase),(hp.T_y2,)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, mags, magmels, realmels, imagmels, freqs, num_batch
            else:
                if hp.predict_world:
                    texts, texts_tests, mels, dones, mags, pitches, harmonics, aperiodics = tf.train.batch([text, texts_test, mel, done, mag, pitch, harmonic, aperiodic],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,), (hp.T_y, 1+hp.n_fft//2),(hp.T_y3,),(hp.T_y3,hp.world_d),(hp.T_y3,hp.world_d)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*1024,
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, mags, pitches, harmonics, aperiodics, num_batch
                else:
                    texts, texts_tests, mels, dones, mags = tf.train.batch([text, texts_test, mel, done, mag],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,), (hp.T_y, 1+hp.n_fft//2)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, mags, num_batch
        else:
            if hp.predict_melograph:
                if hp.predict_world:
                    texts, texts_tests, mels, dones, magmels, realmels, imagmels, freqs, pitches, harmonics, aperiodics = tf.train.batch([text, texts_test, mel, done, magmel, realmel, imagmel, freq, pitch, harmonic, aperiodic],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,),
                                    (hp.T_y2,hp.n_mels),(hp.T_y2,hp.nbins_phase),(hp.T_y2,hp.nbins_phase),(hp.T_y2,),(hp.T_y3,),(hp.T_y3,hp.world_d),(hp.T_y3,hp.world_d)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, magmels, realmels, imagmels, freqs, pitches, harmonics, aperiodics, num_batch
                else:
                    texts, texts_tests, mels, dones, magmels, realmels, imagmels, freqs = tf.train.batch([text, texts_test, mel, done, magmel, realmel, imagmel, freq],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,),
                                    (hp.T_y2,hp.n_mels),(hp.T_y2,hp.nbins_phase),(hp.T_y2,hp.nbins_phase),(hp.T_y2,)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, magmels, realmels, imagmels, freqs, num_batch
            else:
                if hp.predict_world:
                    texts, texts_tests, mels, dones, pitches, harmonics, aperiodics = tf.train.batch([text, texts_test, mel, done, pitch, harmonic, aperiodic],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,),(hp.T_y3,),(hp.T_y3,hp.world_d),(hp.T_y3,hp.world_d)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, pitches, harmonics, aperiodics, num_batch
                else:
                    texts, texts_tests, mels, dones = tf.train.batch([text, texts_test, mel, done],
                                    shapes=[(hp.T_x,), (hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,)],
                                    num_threads=4,
                                    batch_size=hp.batch_size, 
                                    capacity=hp.batch_size*16,   
                                    dynamic_pad=False)
                    return texts_tests, texts, mels, dones, num_batch