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

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_data(config,training=True):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, _texts_test, mels, dones, mags = [], [], [], [], []
    num_samples = 1
    metadata = os.path.join(config.data_paths, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        fname, _, sent = line.strip().split("|")
        sent = text_normalize(sent) + "E" # text normalization, E: EOS
        if len(sent) <= hp.T_x:
            sent += "P"*(hp.T_x-len(sent)) #this was added
            pstring = [char2idx[char] for char in sent]  
            texts.append(np.array(pstring, np.int32).tostring())
            _texts_test.append(np.array(pstring,np.int32))
            mels.append(os.path.join(config.data_paths, "mels", fname + ".npy"))
            dones.append(os.path.join(config.data_paths, "dones", fname + ".npy"))
            mags.append(os.path.join(config.data_paths, "mags", fname + ".npy"))

            # if num_samples==hp.batch_size:
            #     if training: texts, _texts_test, mels, dones, mags = [], [], [], [], []
            #     else: # for evaluation
            #         num_samples += 1
            #         return texts, _texts_test, mels, dones, mags
            # num_samples += 1
    # else: # nick
    #     metadata = os.path.join(hp.data, 'metadata.tsv')
    #     for line in codecs.open(metadata, 'r', 'utf-8'):
    #         fname, sent = line.strip().split("\t")
    #         sent = text_normalize(sent) + "E"  # text normalization, E: EOS
    #         if len(sent) <= hp.Tx:
    #             texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
    #             mels.append(os.path.join(hp.data, "mels", fname.split("/")[-1].replace(".wav", ".npy")))
    #             dones.append(os.path.join(hp.data, "dones", fname.split("/")[-1].replace(".wav", ".npy")))
    #             mags.append(os.path.join(hp.data, "mags", fname.split("/")[-1].replace(".wav", ".npy")))
    #
    #             if num_samples==hp.batch_size:
    #                 if training: texts, mels, dones, mags = [], [], [], []
    #                 else: # for evaluation
    #                     num_samples += 1
    #                     return texts, mels, dones, mags
    #             num_samples += 1
    return texts, _texts_test, mels, dones, mags

def load_test_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        sent = text_normalize(line).strip() + "E" # text normalization, E: EOS
        if len(sent) <= hp.T_x:
            sent += "P"*(hp.T_x-len(sent))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts

def get_batch(config):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _texts_test, _mels, _dones, _mags = load_data(config)

        # Calc total batch count
        num_batch = len(_texts) // hp.batch_size
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        mels = tf.convert_to_tensor(_mels)
        dones = tf.convert_to_tensor(_dones)
        mags = tf.convert_to_tensor(_mags)

        zero_masks = get_zero_masks()
         
        # Create Queues
        text, mel, mel3, done, mag = tf.train.slice_input_producer([texts, mels, mels, dones, mags], shuffle=True)

        # Decoding
        text = tf.decode_raw(text, tf.int32) # (None,)
        mel = tf.py_func(lambda x:np.load(x), [mel], tf.float32) # (None, n_mels)
        done = tf.py_func(lambda x:np.load(x), [done], tf.int32) # (None,)
        mag = tf.py_func(lambda x:np.load(x), [mag], tf.float32) # (None, 1+n_fft/2)

        # Padding
        text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x] # (Tx,)
        mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y] # (Ty, n_mels)
        done = tf.pad(done, ((0, hp.T_y),))[:hp.T_y] # (Ty,)
        mag = tf.pad(mag, ((0, hp.T_y), (0, 0)))[:hp.T_y] # (Ty, 1+n_fft/2)

        # Reduction
        mel = tf.reshape(mel, (hp.T_y//hp.r, -1)) # (Ty/r, n_mels*r)
        if hp.run_pers:
            mel3 = tf.multiply(mel,tf.convert_to_tensor(zero_masks[:,:,randint(0,(hp.T_y//hp.r)//hp.rwin)], np.float32))
        else:
            mel3 = mel
        done = done[::hp.r] # (Ty/r,)

        # create batch queues
        texts, mels, mels2, dones, mags = tf.train.batch([text, mel, mel3, done, mag],
                                shapes=[(hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y//hp.r,), (hp.T_y, 1+hp.n_fft//2)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=False)

    return _texts_test, texts, mels, mels2, dones, mags, num_batch

def get_zero_masks():
    mxval = (hp.T_y//hp.r)//hp.rwin
    zero_masks = []
    mms = []

    for i in range(mxval+1):
        if i == 0:        
            for k in range(0,hp.T_y//hp.r):
                if k == 0:
                    mms = np.zeros(hp.n_mels*hp.r)
                else:
                    mms = np.vstack((mms,np.zeros(hp.n_mels*hp.r)))
        else:
            for k in range(0,i*hp.rwin):
                if k == 0:
                    mms = np.ones(hp.n_mels*hp.r)
                else:
                    mms = np.vstack((mms,np.ones(hp.n_mels*hp.r)))
            for k in range(i*hp.rwin,hp.T_y//hp.r):
                mms = np.vstack((mms,np.zeros(hp.n_mels*hp.r)))
        if i == 0:
            zero_masks = mms
        else:
            zero_masks = np.dstack((zero_masks,mms))

    return tf.convert_to_tensor(np.array(zero_masks),dtype=tf.float32)