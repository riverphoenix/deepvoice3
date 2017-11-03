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
from train import Graph
from utils import spectrogram2wav
from data_load import load_test_data
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

def create_write_files(sess,g,x,mname,cdir,samples):
    # Inference
    mels = np.zeros((len(x), hp.T_y//hp.r, hp.n_mels*hp.r), np.float32)
    prev_max_attentions = np.zeros((len(x),), np.int32)
    for j in range(hp.T_x):
        _mels, _max_attentions = sess.run([g.mels, g.max_attentions],
                                          {g.x: x,
                                           g.y1: mels,
                                           g.prev_max_attentions: prev_max_attentions})
        mels[:, j, :] = _mels[:, j, :]
        prev_max_attentions = _max_attentions[:, j]
    mags = sess.run(g.mags, {g.mels: mels})

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

def synthesize():
    # Load graph
    g = Graph(training=False); print("Graph loaded")
    X = load_test_data()
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
             
            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]

            # Inference
            for i in range(0, len(X), hp.batch_size):
                x = X[i:i+hp.batch_size]

                # Get melspectrogram
                mels = np.zeros((hp.batch_size, hp.T_y//hp.r, hp.n_mels*hp.r), np.float32)
                prev_max_attentions = np.zeros((hp.batch_size,), np.int32)
                for j in range(hp.T_x):
                    _mels, _max_attentions = sess.run([g.mels, g.max_attentions],
                                                      {g.x: x,
                                                       g.y1: mels,
                                                       g.prev_max_attentions: prev_max_attentions})
                    mels[:, j, :] = _mels[:, j, :]
                    prev_max_attentions = _max_attentions[:, j]
                # librosa.display.specshow(librosa.power_to_db(mels[0], ref=np.max))
                # plt.colorbar(format='%+2.0f dB')
                # plt.title('Mel spectrogram')
                # plt.tight_layout()
                # plt.savefig('f.png')
                # plt.show()
                # np.save('temp.npy', mels[0])

                # Get magnitude
                mags = sess.run(g.mags, {g.mels: mels})

    # Generate wav files
    if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
    for i, mag in enumerate(mags):
        # generate wav files
        mag = mag*hp.mag_std + hp.mag_mean # denormalize
        audio = spectrogram2wav(np.power(10, mag))
        write(hp.sampledir + "/{}_{}.wav".format(mname, i), hp.sr, audio)
                # Generate wav files
                if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
                for i, mag in enumerate(mags):
                    # generate wav files
                    mag = mag*hp.mag_std + hp.mag_mean # denormalize
                    audio = spectrogram2wav(np.power(10, mag) * hp.sharpening_factor)
                    write(hp.sampledir + "/{}_{}.wav".format(mname, i+1), hp.sr, audio)
                                          
if __name__ == '__main__':
    synthesize()
    print("Done")
    
    
