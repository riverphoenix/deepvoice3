# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

import os

from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import spectrogram2wav
from data_load import load_test_data
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import argparse

def create_write_files(sess,g,x,mname,cdir,samples):
    # Inference
    decoder_outputs = np.zeros((len(x), hp.T_y//hp.r, hp.attention_size), np.float32) # added decoder_outputs
    mels = np.zeros((len(x), hp.T_y//hp.r, hp.n_mels*hp.r), np.float32)
    prev_max_attentions = np.zeros((len(x),), np.int32)
    for j in range(hp.T_x):
        # Added decoder_outputs
        _decoder_outputs,_mels, _max_attentions = sess.run([g.decoder_outputs,g.mels, g.max_attentions],
                                          {g.x: x,
                                           g.y1: mels,
                                           g.prev_max_attentions: prev_max_attentions})
        decoder_outputs[:, j, :] = _decoder_outputs[:, j, :] #decoder_outputs
        mels[:, j, :] = _mels[:, j, :]
        prev_max_attentions = _max_attentions[:, j]
    mags = sess.run(g.mags, {g.decoder_outputs: decoder_outputs}) # changed from mels to decoder_outputs

    # Generate wav files
    z_list = random.sample(range(0,hp.batch_size),samples)

    for i, mag in enumerate(mags):
        # generate wav files
        if i in z_list:
            mag = mag*hp.mag_std + hp.mag_mean # denormalize
            audio = spectrogram2wav(np.power(10, mag))
            write(cdir + "/{}_{}.wav".format(mname, i), hp.sr, audio)

def synthesize_part(grp,config,gs,x_train):

    x_train = random.sample(x_train, hp.batch_size)
    x_test = load_test_data()

    with grp.graph.as_default():
        sv = tf.train.Supervisor(logdir=config.log_dir)
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))

            create_write_files(sess,grp,x_train,"sample_"+str(gs)+"_train_",config.log_dir,config.train_samples)
            create_write_files(sess,grp,x_test,"sample_"+str(gs)+"_test_",config.log_dir,config.test_samples)

            sess.close()