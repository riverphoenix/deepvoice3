# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_test_data, invert_text
from scipy.io.wavfile import write
import random
import magphase.magphase as mp
import pyworld as pw
import librosa

def create_write_files(ret,sess,g,x,mname,cdir,typeS):

    x = np.expand_dims(x, axis=0)
    # Get melspectrogram
    mel_output = np.zeros((1, hp.T_y // hp.r, hp.n_mels * hp.r), np.float32)
    decoder_output = np.zeros((1, hp.T_y // hp.r, hp.embed_size), np.float32)
    alignments_li = np.zeros((hp.dec_layers, hp.T_x, hp.T_y//hp.r), np.float32)
    prev_max_attentions_li = np.zeros((hp.dec_layers, 1), np.int32)
    for j in range(hp.T_y // hp.r):
        _gs, _mel_output, _decoder_output, _max_attentions_li, _alignments_li = \
            sess.run([g.global_step, g.mel_output, g.decoder_output, g.max_attentions_li, g.alignments_li],
                     {g.x: x,
                      g.y1: mel_output,
                      g.prev_max_attentions_li:prev_max_attentions_li})
        mel_output[:, j, :] = _mel_output[:, j, :]
        decoder_output[:, j, :] = _decoder_output[:, j, :]
        prev_max_attentions_li = np.array(_max_attentions_li)[:, :, j]
       
    # Get magnitude
    if hp.predict_griffin:
        if hp.predict_melograph:
            if hp.predict_world:
                mag_output, magmel_output, realmel_output, imagemel_output, freq_output, pitch_output, harmonic_output, aperiodic_output = sess.run([g.mag_output, g.magmel_output, g.realmel_output, g.imagemel_output, g.freq_output, g.pitch_output, g.harmonic_output, g.aperiodic_output], {g.decoder_output: decoder_output})
            else:
                mag_output, magmel_output, realmel_output, imagemel_output, freq_output = sess.run([g.mag_output, g.magmel_output, g.realmel_output, g.imagemel_output, g.freq_output], {g.decoder_output: decoder_output})
        else:
            if hp.predict_world:
                mag_output, pitch_output, harmonic_output, aperiodic_output = sess.run([g.mag_output, g.pitch_output, g.harmonic_output, g.aperiodic_output], {g.decoder_output: decoder_output})
            else:
                mag_output = sess.run([g.mag_output], {g.decoder_output: decoder_output})
    else:
        if hp.predict_melograph:
            if hp.predict_world:
                magmel_output, realmel_output, imagemel_output, freq_output, pitch_output, harmonic_output, aperiodic_output = sess.run([g.magmel_output, g.realmel_output, g.imagemel_output, g.freq_output, g.pitch_output, g.harmonic_output, g.aperiodic_output], {g.decoder_output: decoder_output})
            else:
                magmel_output, realmel_output, imagemel_output, freq_output = sess.run([g.magmel_output, g.realmel_output, g.imagemel_output, g.freq_output], {g.decoder_output: decoder_output})
        else:
            if hp.predict_world:
                pitch_output, harmonic_output, aperiodic_output = sess.run([g.pitch_output, g.harmonic_output, g.aperiodic_output], {g.decoder_output: decoder_output})
    
    # Generate wav files
    
    x = np.squeeze(x, axis=0)
    txt = invert_text(x)

    if hp.predict_griffin:
        mag_output = np.squeeze(mag_output[0])
        wav = spectrogram2wav(mag_output)
        write(cdir + "/{}_grif.wav".format(mname), hp.sr, wav)
        ret.append([txt,wav,typeS+"_grif"])

    if hp.predict_melograph:
        magmel_output = np.squeeze(magmel_output[0])
        realmel_output = np.squeeze(realmel_output[0])
        imagemel_output = np.squeeze(imagemel_output[0])
        freq_output = np.squeeze(freq_output[0])
        wav = mp.synthesis_from_compressed(magmel_output, realmel_output, imagemel_output, freq_output, hp.sr, hp.n_fft)
        write(cdir + "/{}_melgraph.wav".format(mname), hp.sr, wav)
        ret.append([txt,wav,typeS+"_melgraph"])

    if hp.predict_world:
        pitch_output = np.squeeze(pitch_output[0])
        harmonic_output = np.squeeze(harmonic_output[0])
        aperiodic_output = np.squeeze(aperiodic_output[0])
        wav = pw.synthesize(np.float64(pitch_output), np.float64(harmonic_output), np.float64(aperiodic_output), hp.sr,  pw.default_frame_period)
        wav, _ = librosa.effects.trim(wav)
        write(cdir + "/{}_world.wav".format(mname), hp.sr, wav)
        ret.append([txt,wav,typeS+"_world"])
    
    return ret

def synthesize_part(grp,config,gs,x_train):
    
    x_train = random.sample(x_train, hp.batch_size)
    x_test = load_test_data()
    rand = random.randint(0,hp.batch_size-1)
    x_train = x_train[rand]
    x_test = x_test[rand]
    
    wavs = []
    with grp.graph.as_default():
        sv = tf.train.Supervisor(logdir=config.log_dir)
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))

            wavs = create_write_files(wavs,sess,grp,x_train,"sample_"+str(gs)+"_train_",config.log_dir,"train")
            wavs = create_write_files(wavs,sess,grp,x_test,"sample_"+str(gs)+"_test_",config.log_dir,"test")

            sess.close()
    return wavs

def synthesize():
    # Load data
    X = load_test_data()

    # Load graph
    g = Graph(training=False); print("Graph loaded")

    # Inference
    with g.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()

            # Restore parameters
            saver.restore(sess, tf.train.latest_checkpoint(hp.puresynth)); print("Restored!")

            # Get model name
            mname = open(hp.puresynth + '/checkpoint', 'r').read().split('"')[1]

            # Synthesize
            file_id = 1
            for i in range(0, len(X), hp.batch_size):
                x = X[i:i + hp.batch_size]

                # Get melspectrogram
                mel_output = np.zeros((hp.batch_size, hp.T_y // hp.r, hp.n_mels * hp.r), np.float32)
                decoder_output = np.zeros((hp.batch_size, hp.T_y // hp.r, hp.embed_size), np.float32)
                alignments_li = np.zeros((hp.dec_layers, hp.T_x, hp.T_y//hp.r), np.float32)
                prev_max_attentions_li = np.zeros((hp.dec_layers, hp.batch_size), np.int32)
                for j in range(hp.T_y // hp.r):
                    _gs, _mel_output, _decoder_output, _max_attentions_li, _alignments_li = \
                        sess.run([g.global_step, g.mel_output, g.decoder_output, g.max_attentions_li, g.alignments_li],
                                 {g.x: x,
                                  g.y1: mel_output,
                                  g.prev_max_attentions_li:prev_max_attentions_li})
                    mel_output[:, j, :] = _mel_output[:, j, :]
                    decoder_output[:, j, :] = _decoder_output[:, j, :]
                    alignments_li[:, :, j] = np.array(_alignments_li)[:, :, j]
                    prev_max_attentions_li = np.array(_max_attentions_li)[:, :, j]

                # Get magnitude
                mag_output = sess.run(g.mag_output, {g.decoder_output: decoder_output})

                # Generate wav files
                if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
                for mag in mag_output:
                    print("Working on file num ", file_id)
                    wav = spectrogram2wav(mag)
                    write(hp.sampledir + "/{}_{}.wav".format(mname, file_id), hp.sr, wav)
                    file_id += 1

if __name__ == '__main__':
    synthesize()
    print("Done")
