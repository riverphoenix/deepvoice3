# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from tqdm import tqdm
import argparse
import os
from glob import glob

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import encoder, decoder, converter
import tensorflow as tf
from fn_utils import infolog
import synthesize

class Graph:
    def __init__(self, config=None,training=True):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Data Feeding
            ## x: Text. (N, T_x), int32
            ## y1: Reduced melspectrogram. (N, T_y//r, n_mels*r) float32
            ## y2: Reduced dones. (N, T_y//r,) int32
            ## z: Magnitude. (N, T_y, n_fft//2+1) float32
            if training:
                self.origx, self.x, self.y1, self.y2, self.z, self.num_batch = get_batch(config)
                self.prev_max_attentions = tf.constant([0]*hp.batch_size)
            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.T_x))
                self.y1 = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.T_y//hp.r, hp.n_mels*hp.r))
                self.prev_max_attentions = tf.placeholder(tf.int32, shape=(hp.batch_size,))

            # Get decoder inputs: feed last frames only (N, T_y//r, n_mels)
            self.decoder_inputs = tf.concat((tf.zeros_like(self.y1[:, :1, -hp.n_mels:]), self.y1[:, :-1, -hp.n_mels:]), 1)

            # Networks
            with tf.variable_scope("net"):
                # Encoder. keys: (N, T_x, E), vals: (N, T_x, E)
                self.keys, self.vals, self.masks = encoder(self.x,
                                               training=training,
                                               scope="encoder")

                # Decoder. mels: (N, T_y/r, n_mels*r), dones: (N, T_y/r, 2), alignments: (N, T_y, T_x)
                self.mels, self.dones, self.alignments, self.max_attentions = decoder(self.decoder_inputs,
                                                                                     self.keys,
                                                                                     self.vals,
                                                                                     self.masks,
                                                                                     self.prev_max_attentions,
                                                                                     training=training,
                                                                                     scope="decoder",
                                                                                     reuse=None)
                # Restore shape. mel_inputs: (N, T_y, n_mels)
                self.mel_inputs = tf.reshape(self.mels, (hp.batch_size, hp.T_y, hp.n_mels))
                self.mel_inputs = normalize(self.mel_inputs, type=hp.norm_type, training=training, activation_fn=tf.nn.relu)

                # Converter. mags: (N, T_y//r, (1+n_fft//2)*r)
                self.mags = converter(self.mel_inputs,
                                          training=training,
                                          scope="converter",
                                          reuse=None)
            if training:
                # Loss
                self.loss1_mae = tf.reduce_mean(tf.abs(self.mels - self.y1))
                self.loss1_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dones, labels=self.y2))
                self.loss2 = tf.reduce_mean(tf.abs(self.mags - self.z))
                self.loss = self.loss1_mae + self.loss1_ce + self.loss2

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    grad = tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
                    grad = tf.clip_by_norm(grad, hp.max_grad_norm)
                    self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
                   
                # Summary
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('loss1_mae', self.loss1_mae)
                tf.summary.scalar('loss1_ce', self.loss1_ce)
                tf.summary.scalar('loss2', self.loss2)
                
                self.merged = tf.summary.merge_all()

def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default=hp.logdir)
    parser.add_argument('--log_name', default=hp.logname)
    parser.add_argument('--sample_dir', default=hp.sampledir)
    parser.add_argument('--data_paths', default=hp.data)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--initialize_path', default=None)

    parser.add_argument('--summary_interval', type=int, default=hp.summary_interval)
    parser.add_argument('--test_interval', type=int, default=hp.test_interval)
    parser.add_argument('--checkpoint_interval', type=int, default=hp.checkpoint_interval)
    parser.add_argument('--train_samples', type=int, default=hp.train_iterations)
    parser.add_argument('--test_samples', type=int, default=hp.test_iterations)
    parser.add_argument('--num_iterations', type=int, default=hp.num_iterations)

    config = parser.parse_args()
    
    config.log_dir = config.log_dir + '/' + config.log_name
    if not os.path.exists(config.log_dir): os.makedirs(config.log_dir)

    log = infolog.log
    log_path = os.path.join(config.log_dir, 'train.log')
    infolog.init(log_path, "log")

    checkpoint_path = os.path.join(config.log_dir, 'model.ckpt')

    g = Graph(config=config);
    print("Training Graph loaded")

    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=config.log_dir, save_model_secs=0)
        with sv.managed_session() as sess:

            if config.load_path:
                # Restore from a checkpoint if the user requested it.
                restore_path = get_most_recent_checkpoint(config.load_path)
                sv.saver.restore(sess, restore_path)
                log('Resuming from checkpoint: %s ' % (restore_path), slack=True)
            elif config.initialize_path:
                restore_path = get_most_recent_checkpoint(config.initialize_path)
                sv.saver.restore(sess, restore_path)
                log('Initialized from checkpoint: %s ' % (restore_path), slack=True)
            else:
                log('Starting new training', slack=True)

            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            with open('temp.txt', 'w') as fout:
                for epoch in range(1, 100000000):
                    if sv.should_stop(): break
                    for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                        sess.run(g.train_op)

                    gs = sess.run(g.global_step)

                    if epoch % config.summary_interval == 0:
                        log('Writing summary at step: %d' % gs)
                        summary_writer.add_summary(sess.run(g.merged),gs)

                    if epoch % config.checkpoint_interval == 0:
                        log('Saving checkpoint to: %s-%d' % (checkpoint_path, gs))
                        sv.saver.save(sess, checkpoint_path, global_step=g.global_step)

                    if epoch % config.test_interval == 0:
                        log('Saving audio and alignment...')
                        synthesize.synthesize_part(g,sess,config)

                    # break
                    if gs > config.num_iterations: break

    print("Done")

if __name__ == '__main__':
    main()
