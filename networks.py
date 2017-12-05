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

def encoder(inputs, training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, Tx], with dtype of int32. Encoder inputs.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, Tx, e).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("text_embedding"):
            embedding = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, Tx, e)

        with tf.variable_scope("encoder_prenet"):
            tensor = fc_block(embedding, hp.enc_channels, training=training) # (N, Tx, c)

        with tf.variable_scope("encoder_conv"):
            for i in range(hp.enc_layers):
                outputs = conv_block(tensor,
                                    size=hp.enc_filter_size,
                                    rate=2**i,
                                    training=training,
                                    scope="encoder_conv_{}".format(i)) # (N, Tx, c)
                tensor = (outputs + tensor) * tf.sqrt(0.5)

        with tf.variable_scope("encoder_postnet"):
            keys = fc_block(tensor, hp.embed_size, training=training) # (N, Tx, e)
            vals = tf.sqrt(0.5) * (keys + embedding) # (N, Tx, e)

    return keys, vals

def decoder(inputs,
            keys,
            vals,
            prev_max_attentions_li=None,
            training=True,
            scope="decoder",
            reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, Ty/r, n_mels]. Shifted log melspectrogram of sound files.
      keys: A 3d tensor with shape of [N, Tx, e].
      vals: A 3d tensor with shape of [N, Tx, e].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    if training:
        bc_batch = hp.batch_size
    else:
        bc_batch = 1

    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("decoder_prenet"):
            for i in range(hp.dec_layers):
                inputs = fc_block(inputs,
                                  num_units=hp.embed_size,
                                  dropout_rate=0 if i==0 else hp.dropout_rate,
                                  activation_fn=tf.nn.relu,
                                  training=training,
                                  scope="decoder_prenet_{}".format(i)) # (N, Ty/r, a)

        with tf.variable_scope("decoder_conv_att"):
            with tf.variable_scope("positional_encoding"):
                if hp.sinusoid:
                    query_pe = positional_encoding(inputs[:, :, 0],
                                                   num_units=hp.embed_size,
                                                   position_rate=1.,
                                                   zero_pad=False,
                                                   scale=True)  # (N, Ty/r, e)
                    key_pe = positional_encoding(keys[:, :, 0],
                                                num_units=hp.embed_size,
                                                position_rate=(hp.T_y // hp.r) / hp.T_x,
                                                zero_pad=False,
                                                scale=True)  # (N, Tx, e)
                else:
                    query_pe = embed(tf.tile(tf.expand_dims(tf.range(hp.T_y // hp.r), 0), [bc_batch, 1]),
                             vocab_size=hp.T_y,
                             num_units=hp.embed_size,
                             zero_pad=False,
                             scope="query_pe")

                    key_pe = embed(tf.tile(tf.expand_dims(tf.range(hp.T_x), 0), [bc_batch, 1]),
                                  vocab_size=hp.T_x,
                                  num_units=hp.embed_size,
                                  zero_pad=False,
                                  scope="key_pe")

            alignments_li, max_attentions_li = [], []
            for i in range(hp.dec_layers):
                _inputs = inputs
                queries = conv_block(inputs,
                                     size=hp.dec_filter_size,
                                     rate=2**i,
                                     padding="CAUSAL",
                                     training=training,
                                     scope="decoder_conv_block_{}".format(i)) # (N, Ty/r, a)

                inputs = (queries + inputs) * tf.sqrt(0.5)

                # residual connection
                queries = inputs + query_pe
                keys += key_pe

                # Attention Block.
                # tensor: (N, Ty/r, e)
                # alignments: (N, Ty/r, Tx)
                # max_attentions: (N, Ty/r)
                tensor, alignments, max_attentions = attention_block(queries,
                                                                     keys,
                                                                     vals,
                                                                     dropout_rate=hp.dropout_rate,
                                                                     prev_max_attentions=prev_max_attentions_li[i],
                                                                     mononotic_attention=(not training and i>2),
                                                                     training=training,
                                                                     scope="attention_block_{}".format(i))

                inputs = (tensor + queries) * tf.sqrt(0.5)
                    # inputs = (inputs + _inputs) * tf.sqrt(0.5)
                alignments_li.append(alignments)
                max_attentions_li.append(max_attentions)

        decoder_output = inputs

        with tf.variable_scope("mel_logits"):
            mel_logits = fc_block(decoder_output, hp.n_mels*hp.r, training=training)  # (N, Ty/r, n_mels*r)

    return mel_logits, decoder_output, alignments_li, max_attentions_li

def converter(inputs, inputs_back, training=True, scope="converter", reuse=None):
    '''Converter
    Args:
      inputs: A 3d tensor with shape of [N, Ty, v]. Activations of the reshaped outputs of the decoder.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''

    if training:
        bc_batch = hp.batch_size
    else:
        bc_batch = 1

    world_input = inputs_back
    world_input2 = inputs_back

    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("converter_conv_world"):
            for i in range(hp.converter_layers):
                world_outputs = conv_block(world_input,
                                     size=hp.converter_filter_size,
                                     rate=2**i,
                                     padding="SAME",
                                     training=training,
                                     scope="converter_conv_{}".format(i))  
                world_input = (world_input + world_outputs) * tf.sqrt(0.5)

        with tf.variable_scope("world_logits_fc"):
            # magphase_logits_fc = fc_block(magphase_input, hp.converter_channels, activation_fn=tf.nn.relu, training=training)
            world_logits_fc = fc_block(world_input, hp.embed_size, training=training)
            #world_logits_fc = world_input

        ########### upsample ###############
        world_logits_fc = tf.expand_dims(world_logits_fc, -1)
        world_logits_up = tf.image.resize_nearest_neighbor(world_logits_fc, [hp.T_y2,world_logits_fc.get_shape()[2]])
        world_logits_up = tf.squeeze(world_logits_up,-1)

        with tf.variable_scope("world_logits_conv"):
          world_logits_conv = conv_block(world_logits_up,
                                     size=hp.converter_filter_size,
                                     rate=1,
                                     padding="SAME",
                                     training=training,
                                     scope="converter_conv_magphase_{}".format(0))  
          world_logits_conv = (world_logits_up + world_logits_conv) * tf.sqrt(0.5)

        with tf.variable_scope("harmonic_logits_fc"):
            harmonic_logits = fc_block(world_logits_conv, hp.world_d, training=training)

        with tf.variable_scope("pitch_logits_fc"):
            pitch_logits = fc_block(world_logits_conv, 1, training=training)
            pitch_logits = tf.squeeze(pitch_logits,-1)

        with tf.variable_scope("aperiodic_logits_fc"):
            aperiodic_logits = fc_block(world_logits_conv, hp.world_d, training=training)            

    return pitch_logits, harmonic_logits, aperiodic_logits