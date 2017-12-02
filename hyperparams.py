# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
import math

def get_T_y(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 22050 # Sampling rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds 0.0125
    frame_length = 0.05 # seconds 0.05
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    nbins_phase = 60
    world_d = 513
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = 0.97 # or None 0.97
    max_db = 100
    ref_db = 20
    dropout_rate = .2 # .05
    norm_type = "ins" # TODO: weight normalization

    # Model
    r = 4 # Reduction factor 4
    run_cmu = True
    sinusoid = False
    
    predict_griffin = True
    predict_melograph = False
    predict_world = True
    

    ## Enocder
    if not run_cmu:
        vocab_size = 32
    else:
        vocab_size = 53
    embed_size = 256 # == e
    enc_layers = 7
    enc_filter_size = 5
    enc_channels = 64 # == c 256
    ## Decoder
    dec_layers = 4
    dec_filter_size = 5
    attention_size = 128*2 # == a 128
    ## Converter
    converter_layers = 5*2
    converter_filter_size = 5
    converter_channels = 256 # == v
    attention_win_size = 3
	
    # data
    max_duration = 10.0#10.10 # seconds
    T_x = 180 #200 # characters. maximum length of text.
    T_y = int(get_T_y(max_duration, sr, hop_length, r)) # Maximum length of sound (frames)
    T_y2 = 3* T_y
    T_y3 = 3* T_y

    # training scheme
    optim = 'adam'
    lr = 0.001
    logdir = "logs"
    logname = 'demos'
    sampledir = 'samples'
    puresynth = 'logs/first2'
    batch_size = 16
    max_grad_norm = 100.
    max_grad_val = 5.
    num_iterations = 500000

    # Prepo params
    data = 'datasets/LJMag'
    prepro_gpu = 8
    create_melograph = True
    create_world = True

    # Training and Testing

    summary_interval = 1
    test_interval = 1
    checkpoint_interval = 1

    # fix generation of magphase
    # fix length of world
    # fix save checkpoint when griffin not used
