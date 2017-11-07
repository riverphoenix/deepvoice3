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
import pandas as pd

cmu = pd.read_csv('cmudict.dict.txt',header=None,names=['name'])
cmu['word'], cmu['phone'] = cmu['name'].str.split(' ', 1).str
cmu['word'] = cmu['word'].str.upper()
cmu.drop(['name'],axis=1,inplace=True)
cmu = list(cmu.set_index('word').to_dict().values()).pop()

def text_normalize(sent):
    '''Remove accents and upper strings.'''
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')
    normalized = re.sub("[^ A-Z',;.]", "", _strip_accents(sent).upper())
    if normalized[-1] in [".",",","?",";"]:
        normalized = normalized[0:-1]
    normalized = re.sub('\'',' ',normalized)
    normalized = re.sub(' ','@',normalized)
    normalized = re.sub(',','@@',normalized)
    normalized = re.sub(';','@@@',normalized)
    normalized = re.sub('\.','@@@@',normalized)
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
    valid_symbols = ['@','A','AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'C','CH', 'D', 'DH', 'E','EH', 'ER', 'EY',
    'F', 'G', 'H','HH', 'I','IH', 'IY', 'J','JH', 'K', 'L', 'M', 'N', 'NG', 'OW','O', 'OY', 'P', 'Q','R', 'S', 'SH',
    'T', 'TH', 'U','UH', 'UW','V', 'W', 'X','Y', 'Z', 'ZH','*',"'"]
    _valid_symbol_set = set(valid_symbols)
    
    char2idx = {char: idx for idx, char in enumerate(_valid_symbol_set)}
    idx2char = {idx: char for idx, char in enumerate(_valid_symbol_set)}
    
    return char2idx, idx2char

def str_to_idx(strin,char2idx):
    strin = re.sub('([A-Z])@','\\1 @',strin)
    strin = re.sub('([A-Z])\*','\\1 *',strin)
    strin = re.sub('@([A-Z])','@ \\1',strin)
    strin = re.sub('@',' @',strin)
    strin = re.sub("\\s+", " ",strin)
    strin = re.split('\s',strin)
    rtstr = [char2idx[char] for char in strin]
    return rtstr

def load_train_data(config):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, texts_test, mels, dones, mags = [], [], [], [], []
    metadata = os.path.join(config.data_paths, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        fname, _, sent = line.strip().split("|")

        sent = text_normalize(sent) + "*" # text normalization, *: EOS
        sent = break_to_phonemes(sent)
        if len(sent) <= hp.T_x:
            sent += "@"*(hp.T_x-len(sent))
            pstring = str_to_idx(sent,char2idx)
            texts.append(np.array(pstring, np.int32).tostring())
            texts_test.append(np.array(pstring,np.int32))
            mels.append(os.path.join(config.data_paths, "mels", fname + ".npy"))
            dones.append(os.path.join(config.data_paths, "dones", fname + ".npy"))
            mags.append(os.path.join(config.data_paths, "mags", fname + ".npy"))
    return texts, texts_test, mels, dones, mags

def load_test_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        sent = text_normalize(sent) + "*" # text normalization, *: EOS
        sent = break_to_phonemes(sent)
        if len(sent) <= hp.T_x:
            sent += "@"*(hp.T_x-len(sent))
            pstring = str_to_idx(sent,char2idx)
            texts.append(np.array(pstring,np.int32))
    return texts

def get_batch(config):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _texts_test, _mels, _dones, _mags = load_train_data(config) # bytes

        # Calc total batch count
        num_batch = len(_texts) // hp.batch_size
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        mels = tf.convert_to_tensor(_mels)
        dones = tf.convert_to_tensor(_dones)
        mags = tf.convert_to_tensor(_mags)
         
        # Create Queues
        text, mel, done, mag = tf.train.slice_input_producer([texts, mels, dones, mags], shuffle=True)

        # Decoding.
        text = tf.decode_raw(text, tf.int32) # (Tx,)
        mel = tf.py_func(lambda x:np.load(x), [mel], tf.float32) # (Ty/r, n_mels*r)
        done = tf.py_func(lambda x:np.load(x), [done], tf.int32) # (Ty,)
        mag = tf.py_func(lambda x:np.load(x), [mag], tf.float32) # (Ty, 1+n_fft/2)

        # create batch queues
        texts, mels, dones, mags = tf.train.batch([text, mel, done, mag],
                                shapes=[(hp.T_x,), (hp.T_y, hp.n_mels), (hp.T_y,), (hp.T_y, 1+hp.n_fft//2)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=False)

    return _texts_test, texts, mels, dones, mags, num_batch