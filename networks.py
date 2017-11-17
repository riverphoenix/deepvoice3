# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

def encoder(inputs, training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x], with dtype of int32. Encoder inputs.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    masks = tf.sign(tf.abs(inputs))  # (N, T_x)
    with tf.variable_scope(scope, reuse=reuse):
        # Text Embedding
        embedding = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, T_x, E)

        # Encoder PreNet
        tensor = fc_block(embedding,
                          num_units=hp.enc_channels,
                          dropout_rate=hp.dropout_rate_enc,
                          norm_type=hp.norm_type,
                          activation_fn=eval(hp.fc_enc_activ_fn), # changed to relu
                          training=training,
                          scope="prenet_fc_block") # (N, T_x, c)
        
        # Convolution Blocks
        for i in range(hp.enc_layers):
            tensor = conv_block(tensor,
                                size=hp.enc_filter_size,
                                norm_type=hp.norm_type,
                                activation_fn=glu,
                                training=training,
                                scope="encoder_conv_block_{}".format(i)) # (N, T_x, c)

        # Encoder PostNet
        keys = fc_block(tensor,
                        num_units=hp.embed_size,
                        dropout_rate=hp.dropout_rate_enc,
                        norm_type=hp.norm_type,
                        activation_fn=eval(hp.fc_enc_activ_fn),  #changed to relu
                        training=training,
                        scope="postnet_fc_block") # (N, T_x, E)
        vals = tf.sqrt(0.5) * (keys + embedding) # (N, T_x, E)

    return keys, vals, masks, embedding

def decoder_multi(inputs,
            keys,
            vals,
            masks,
            prev_max_attentions=None,
            training=True,
            scope="decoder",
            reuse=None):

  #for zrw in range(((hp.T_y//hp.r)//hp.rwin)+1):
  for zrw in range(1):
    mel_output, done_output, decoder_output, alignments, max_attentions = decoder(inputs,
      keys,vals,masks, prev_max_attentions,training=training,scope=scope+"_"+str(zrw),reuse=None)
    inputs = tf.concat((mel_output[:,:hp.rwin*zrw,:],tf.zeros_like(mel_output[:, hp.rwin*zrw:, :])),1)
    prev_max_attentions = max_attentions

  return mel_output, done_output, decoder_output, alignments, max_attentions

def decoder(inputs,
            keys,
            vals,
            masks,
            prev_max_attentions=None,
            training=True,
            scope="decoder",
            reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Shifted log melspectrogram of sound files.
      keys: A 3d tensor with shape of [N, T_x, E].
      vals: A 3d tensor with shape of [N, T_x, E].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):

        
        #Add a GRU layer
        with tf.variable_scope('GRU_layer', reuse=reuse):
          input_lengths = tf.constant(int(hp.T_y/hp.r), shape=[hp.batch_size],dtype=tf.int32)
          outputs, states = tf.nn.bidirectional_dynamic_rnn( GRUCell(hp.n_mels*hp.r/2), 
            GRUCell(hp.n_mels*hp.r/2), inputs, sequence_length=input_lengths,dtype=tf.float32)
          inputs = tf.concat(outputs,axis=2)

        # Decoder PreNet. inputs:(N, T_y/r, d)
        for i in range(hp.dec_layers):
            inputs = fc_block(inputs,
                              num_units=hp.embed_size,
                              dropout_rate=0 if i==0 else hp.dropout_rate,
                              norm_type=hp.norm_type,
                              activation_fn=tf.nn.relu,
                              training=training,
                              scope="prenet_fc_block_{}".format(i))
        alignments_li, max_attentions_li = [], []
        for i in range(hp.dec_layers):
            # Causal Convolution Block. queries: (N, T_y/r, d)
            queries = conv_block(inputs,
                                 size=hp.dec_filter_size,
                                 padding="CAUSAL",
                                 norm_type=hp.norm_type,
                                 activation_fn=glu,
                                 training=training,
                                 scope="decoder_conv_block_{}".format(i))

            # Attention Block. tensor: (N, T_y/r, d), alignments: (N, T_y, T_x)
            tensor, alignments, max_attentions = attention_block(queries,
                                                                 keys,
                                                                 vals,
                                                                 num_units=hp.attention_size,
                                                                 dropout_rate=hp.dropout_rate,
                                                                 prev_max_attentions=prev_max_attentions[i],
                                                                 training=training,
                                                                 scope="attention_block_{}".format(i))

            # inputs = tensor + queries
            alignments_li.append(alignments)
            max_attentions_li.append(max_attentions)
            inputs = tf.sqrt(0.5) * (tensor + queries) # (N, T_x, E) # missing the sqrt multiplication of this
           
        decoder_output = inputs

        # Readout layers: mel_output: (N, T_y/r, n_mels*r)
        mel_output = fc_block(inputs,
                        num_units=hp.n_mels*hp.r,
                        dropout_rate=hp.dropout_rate2,
                        norm_type=hp.norm_type,
                        activation_fn=eval(hp.fc_dec_activ_fn), # changed to relu
                        training=training,
                        scope="mels")  # (N, T_y/r, n_mels*r*2)

        #mel_output = glu(mel_output)

        ## done_output: # (N, T_y/r, 2)
        done_output = fc_block(inputs,
                         num_units=2,
                         dropout_rate=hp.dropout_rate2,
                         norm_type=hp.norm_type,
                         activation_fn=eval(hp.fc_dec_activ_fn), # changed to relu
                         training=training,
                         scope="dones")

    return mel_output, done_output, decoder_output, alignments_li, max_attentions_li

def converter(inputs, training=True, scope="converter", reuse=None):
    '''Converter
    Args:
      inputs: A 3d tensor with shape of [N, T_y, e/r]. Activations of the reshaped outputs of the decoder.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(hp.converter_layers):
            inputs = conv_block(inputs,
                                 size=hp.converter_filter_size,
                                 padding="SAME",
                                 norm_type=hp.norm_type,
                                 activation_fn=glu,
                                 training=training,
                                 scope="converter_conv_block_{}".format(i))  # (N, T_y/r, d)

        # # Readout layer. mag_output: (N, T_y, n_fft/2+1)
        # mag_output = fc_block(inputs,
        #                num_units=hp.n_fft//2+1,
        #                dropout_rate=hp.dropout_rate2,
        #                norm_type=hp.norm_type,
        #                activation_fn=eval(hp.fc_conv_activ_fn), # changed to relu
        #                training=training,
        #                scope="mag")

        # Added two for mags instead
        mag1 = fc_block(inputs,
                       num_units=hp.n_mels*hp.r,
                       dropout_rate=hp.dropout_rate2,
                       norm_type=hp.norm_type,
                       activation_fn=eval(hp.fc_conv_activ_fn),
                       training=training,
                       scope="mag1")  # (N, T_y/r, 2)

        mag1 = tf.reshape(mag1, (hp.batch_size, hp.T_y, hp.n_mels))

        mag_output = fc_block(mag1,
                       num_units=hp.n_fft//2+1,
                       dropout_rate=hp.dropout_rate2,
                       norm_type=hp.norm_type,
                       activation_fn=eval(hp.fc_conv_activ_fn),
                       training=training,
                       scope="mag2")  # (N, T_y/r, 2)

    return mag_output
