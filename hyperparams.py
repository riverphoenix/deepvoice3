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
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = 0.97 # or None 0.97
    max_db = 100
    ref_db = 20

    # Model
    norm_type = "ins" # TODO: weight normalization
    r = 1 # Reduction factor 4
    run_cmu = True
    sinusoid = False
    normalize_model = False
    predict_griffin = True
    predict_melograph = True
    predict_world = True
    dropout_rate = .2 # .05

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

    train_iterations = 1
    test_iterations = 1

    #Off the shelf
    #Global Step 25 (1350): Loss = 0.11636174 Loss1_mae = 0.06290965 Loss1_ce = 0.00366397 Loss2 = 0.04978812
    #Noise sound
    #Off the shelf remove normilazer from fc and conv
    #Global Step 25 (1350): Loss = 0.10964647 Loss1_mae = 0.05855675 Loss1_ce = 0.00288924 Loss2 = 0.04820048
    #Noise sound    

    #Pers window 4
    #Global Step 25 (1350): Loss = 0.13250279 Loss1_mae = 0.06017732 Loss1_ce = 0.01806482 Loss2 = 0.05426065
    #Global Step 40 (2160): Loss = 0.15257074 Loss1_mae = 0.06233892 Loss1_ce = 0.03070003 Loss2 = 0.05953179
    #Small noise

    #Pers window 4 remove normilazer from fc and conv
    #Global Step 25 (1350): Loss = 0.09492977 Loss1_mae = 0.04700819 Loss1_ce = 0.00124689 Loss2 = 0.04667469
    #Noise but might be (one empty)

    #Pers window 8 remove normilazer from fc and conv
    #Global Step 25 (1350): Loss = 0.16058554 Loss1_mae = 0.05809595 Loss1_ce = 0.04192149 Loss2 = 0.06056810
    #Global Step 40 (2160): Loss = 0.13492824 Loss1_mae = 0.05392413 Loss1_ce = 0.02699178 Loss2 = 0.05401233
    #short noise, no sound on train

    #Off the shelf Window 4
    #Global Step 25 (1350): Loss = 0.16694964 Loss1_mae = 0.08454253 Loss1_ce = 0.02629825 Loss2 = 0.05610886
    #Global Step 47 (2538): Loss = 0.15125100 Loss1_mae = 0.08238565 Loss1_ce = 0.01959107 Loss2 = 0.04927428
    #Noise with norm and without

    #Off the shelf sinusoid
    #Global Step 25 (1350): Loss = 0.16132733 Loss1_mae = 0.08365148 Loss1_ce = 0.02483676 Loss2 = 0.05283908
    #Pers window 1 remove normilazer from fc and conv
    #Global Step 25 (1350): Loss = 0.14150960 Loss1_mae = 0.05494741 Loss1_ce = 0.02942976 Loss2 = 0.05713243



    ##################################
    #Test 1: Off the shelf [rwin=1, norm=True, pers=False, cmu=False]
    #Test 2: Off the shelf [rwin=1, norm=False, pers=False, cmu=False]
    #Test 3: Off the shelf [rwin=1, norm=False, pers=False, cmu=True]

    #check input and how is fed and also on creations