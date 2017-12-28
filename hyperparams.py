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
    world_d = 513
    world_period = 5.0
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 200 # Number of inversion iterations
    preemphasis = 0.97 # or None 0.97
    max_db = 100 #25mel 46mag
    ref_db = 20 #--34mel 30mag
    dropout_rate = .05 # .05 CHANGED TO .05
    norm_type = "ins" # TODO: weight normalization
    lowcut = 70.0
    highcut = 7000.0
    max_db_mel = 30
    ref_db_mel = -80
    max_db_mag = 50
    ref_db_mag = -60

    # Model
    r = 1 # Reduction factor 4 CHANGED TO 1 
    run_cmu = False
    sinusoid = False
    normalization  = True
    test_graph = False
    
    ## Enocder
    phon_drop = 0.2 # USING 0.2 PHON
    if not run_cmu:
        vocab_size = 32
    else:
        vocab_size = 53
    embed_size = 512 # == e 256 CHANGED TO 512
    enc_layers = 7 # 7
    enc_filter_size = 5 # 5
    enc_channels = 64 # == c 64
    ## Decoder
    dec_layers = 4 # 4
    dec_filter_size = 5 # 5
    attention_size = 128 # == a 256 CHANGED TO 128
    
    ## Converter
    converter_layers = 10  # 10
    converter_filter_size = 5  # 5
    
    converter_channels = 256 # == v
    attention_win_size = 3
	
    # data
    max_duration = 10.0#10.10 # seconds
    T_x = 180 #200 # characters. maximum length of text.
    T_y = int(get_T_y(max_duration, sr, hop_length, r)) # Maximum length of sound (frames)
    T_y2 = 3* T_y

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
    data = 'datasets/meltomag'
    prepro_gpu = 16
    # Training and Testing

    summary_interval = 1
    test_interval = 2
    checkpoint_interval = 1000

    # Use other vocoder like Wavenet or train one externally with GRU etc layers on mel to mag network
    # Train on Multispeaker the vocoder only

    #Encoder
    # dropout_rate = .05 # .05 CHANGED TO .05
    # run_cmu = True
    # sinusoid = False
    # normalization  = True
    # Newmodules
    # r = 1
    # phon_drop = 0.2
    # embed_size = 512
    # attention_size = 128
    # preemphasis = 0.97
    # Normalize with ref_db and max_db (change for other sets)
    # lowcut = 70.0
    # highcut = 7000.0

    # Load file -> pre-emphasis -> mel with low/high band -> Normalize
    # Denormalize -> wav generation (griffin_lim) -> pre-emphasis -> low/high band filter (perhaps switch low/high and pre-emphasis)
    # Find better way to normalize

    #meltomag_BILTSMTOCONVADD best for decoder
    #LSTM->BIDRECTIONALDRNN -> IN+OUT*SQR(0.5) -> BIDRECTIONALDRNN -> IN+OUT*SQR(0.5) -> CONV_BLOCK -> THEN (A +B) * SQRT(0.5) -> FC_BLOCK