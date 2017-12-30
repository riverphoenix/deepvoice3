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
from utils import *
from tensorflow.python import debug as tf_debug


class Graph:
    def __init__(self, config=None,training=True):
        # Load vocabulary
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Data Feeding
            ## x: Text. (N, T_x), int32
            ## y1: Reduced melspectrogram. (N, T_y//r, n_mels*r) float32
            ## y2: Reduced dones. (N, T_y//r,) int32
            ## z: Magnitude. (N, T_y, n_fft//2+1) float32
            self.mel = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.T_y//hp.r, hp.n_mels*hp.r),name="mel_inp_pl")
            if training:
                #self.origx, self.x, self.y1, self.y2, self.y3, self.num_batch = get_batch(config)
                self.mel, self.y, self.num_batch = get_batch(config)
                #self.prev_max_attentions_li = tf.ones(shape=(hp.dec_layers, hp.batch_size), dtype=tf.int32)

            with tf.variable_scope("converter"):
                # Converter
                self.mag_logits = converter(self.mel, training=training)
                self.mag_output = tf.nn.sigmoid(self.mag_logits,name="mag_out")
            
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if training:
                self.loss = tf.reduce_mean(tf.abs(self.mag_output - self.y))


                # Training Scheme
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    grad = grad if grad is None else tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
                    grad = grad if grad is None else tf.clip_by_norm(grad, hp.max_grad_norm)
                    self.clipped.append((grad, var))

                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
                
                tf.summary.scalar('loss', self.loss)
              
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
    parser.add_argument('--deltree', default=False)

    parser.add_argument('--summary_interval', type=int, default=hp.summary_interval)
    parser.add_argument('--test_interval', type=int, default=hp.test_interval)
    parser.add_argument('--checkpoint_interval', type=int, default=hp.checkpoint_interval)
    parser.add_argument('--num_iterations', type=int, default=hp.num_iterations)

    parser.add_argument('--debug',type=bool,default=False)

    config = parser.parse_args()
    config.log_dir = config.log_dir + '/' + config.log_name
    if not os.path.exists(config.log_dir): 
        os.makedirs(config.log_dir)
    elif config.deltree:
        for the_file in os.listdir(config.log_dir):
            file_path = os.path.join(config.log_dir, the_file)
            os.unlink(file_path)
    log_path = os.path.join(config.log_dir+'/', 'train.log')
    infolog.init(log_path, "log")
    checkpoint_path = os.path.join(config.log_dir, 'model.ckpt')
    g = Graph(config=config);
    print("Training Graph loaded")
    if hp.test_graph:
        g2 = Graph(config=config,training=False)
        print("Testing Graph loaded")
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=config.log_dir)
        with sv.managed_session() as sess:

            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if config.load_path:
                # Restore from a checkpoint if the user requested it.
                tf.reset_default_graph()
                restore_path = get_most_recent_checkpoint(config.load_path)
                sv.saver.restore(sess, restore_path)
                infolog.log('Resuming from checkpoint: %s ' % (restore_path), slack=True)
            elif config.initialize_path:
                restore_path = get_most_recent_checkpoint(config.initialize_path)
                sv.saver.restore(sess, restore_path)
                infolog.log('Initialized from checkpoint: %s ' % (restore_path), slack=True)
            else:
                infolog.log('Starting new training', slack=True)

            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
            
            for epoch in range(1, 100000000):
                if sv.should_stop(): break
                #losses = [0,0,0,0]
                loss = 0
                for step in tqdm(range(g.num_batch)):
                #for step in range(g.num_batch):
                    #gs,merged,loss,loss1,loss2,loss3,alginm,_ = sess.run([g.global_step,g.merged,g.loss,g.loss1,g.loss2,g.loss3, g.alignments_li,g.train_op])
                    gs,merged,loss_,_ = sess.run([g.global_step,g.merged,g.loss, g.train_op])
                    #loss_one = [loss,loss1,loss2,loss3]
                    loss = loss + loss_
                    #losses = [x + y for x, y in zip(losses, loss_one)]

                #losses = [x / g.num_batch for x in losses]
                loss = loss / g.num_batch
                print("###############################################################################")
                #infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss2 = %.8f Loss3 = %.8f" %(epoch,gs,losses[0],losses[1],losses[2],losses[3]))
                infolog.log("Global Step %d (%04d): Loss = %.8f" %(epoch,gs,loss))
                print("###############################################################################")

                if epoch % config.summary_interval == 0:
                    summary_writer.add_summary(merged,gs)
                    mel, mag, y = sess.run([g.mel, g.mag_output, g.y])
                    plot_loss(config,mag,y,gs)
                    wav_mag, wav_y = get_wavs(config,mag,y,gs)
                    plot_wavs(config,wav_mag,wav_y,gs)
                    if epoch % config.test_interval == 0:
                        save_wavs(config,wav_mag,wav_y,gs)
                        

                # break
                if gs > config.num_iterations: break

    print("Done")

if __name__ == '__main__':
    main()