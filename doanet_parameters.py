# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        #dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='/scratch/asignal/sharath/DCASE2020_SELD_dataset/',

        # OUTPUT PATH
        #feat_label_dir='DCASE2020_SELD_dataset/feat_label/',  # Directory to dump extracted features and labels
        feat_label_dir='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label/',  # Directory to dump extracted features and labels

        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        # DNN MODEL PARAMETERS
        use_hnet=True,
        label_sequence_length=50,    # Feature sequence length
        batch_size=64,              # Batch size
        dropout_rate=0.4,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=128,           # Number of CNN nodes, constant for each layer
        f_pool_size=[2, 2, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,        # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_fnn_layers=2,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes
        nb_epochs=500,               # Train for maximum epochs
        lr=1e-3,
        branch_weights=[1, 1],
        use_dmotp_only=False,
        binary_da=False,
        shuffle_regressors=False
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '50':
        params['use_dmotp_only']= True
        params['shuffle_regressors']=True

    elif argv == '51':
        params['use_dmotp_only']= True
        params['shuffle_regressors']=False

    elif argv == '52':
        params['use_dmotp_only']= False
        params['shuffle_regressors']=True

    elif argv == '53':
        params['use_dmotp_only']= False
        params['shuffle_regressors']= False
    
    elif argv == '54':
        params['batch_size']= 256
        params['label_sequence_length']= 60

    elif argv == '55':
        params['batch_size']= 128
        params['label_sequence_length']= 60

    elif argv == '56':
        params['batch_size']= 64
        params['label_sequence_length']= 60

    elif argv == '57':
        params['batch_size']= 32
        params['label_sequence_length']= 60

    elif argv == '58':
        params['batch_size']= 128
        params['label_sequence_length']= 120

    elif argv == '59':
        params['batch_size']= 128
        params['label_sequence_length']= 30

    elif argv == '60':
        params['batch_size']= 128
        params['label_sequence_length']= 10

    elif argv == '61':
        params['lr']= 1e-2

    elif argv == '62':
        params['lr']= 1e-4
    
    elif argv == '63':
        params['lr']= 1e-5
    

    elif argv == '64':
        params['batch_size']= 64
        params['label_sequence_length']= 15

    elif argv == '65':
        params['batch_size']= 64
        params['label_sequence_length']= 30

    elif argv == '66':
        params['batch_size']= 64
        params['label_sequence_length']= 45

    elif argv == '67':
        params['batch_size']= 64
        params['label_sequence_length']= 60

    elif argv == '68':
        params['lr']= 1e-2
        params['dropout_rate'] = 0.5

    elif argv == '69':
        params['lr']= 1e-3
        params['dropout_rate'] = 0.5

    elif argv == '70':
        params['lr']= 1e-4
        params['dropout_rate'] = 0.5
    
    elif argv == '71':
        params['lr']= 1e-5
        params['dropout_rate'] = 0.5
    
    elif argv == '72':
       params['nb_fnn_layers']=2

    elif argv == '73':
        params['nb_cnn2d_filt'] = 256
        params['rnn_size'] = 256
        params['f_pool_size'] = [4, 2, 2]


    elif argv == '74':
        params['nb_cnn2d_filt'] = 256
        params['rnn_size'] = 256
        params['f_pool_size'] = [2, 2, 2]

    elif argv == '75':
        params['nb_cnn2d_filt'] = 256
        params['rnn_size'] = 512
        params['f_pool_size'] = [2, 2, 2]

    elif argv == '76':
        params['nb_cnn2d_filt'] = 512
        params['rnn_size'] = 512
        params['f_pool_size'] = [2, 2, 2]

    elif argv == '77':
        params['nb_cnn2d_filt'] = 256
        params['rnn_size'] = 256
        params['f_pool_size'] = [1, 1, 1]

    elif argv == '78':
        params['nb_cnn2d_filt'] = 256
        params['rnn_size'] = 512
        params['f_pool_size'] = [1, 1, 1]

    elif argv =='79':
        params['use_hnet']=False

    elif argv =='80':
        params['binary_da']=True

    elif argv =='81':
        params['self_attn']=True

    elif argv == '12':
        params['nb_cnn2d_filt'] = 32
        params['rnn_size'] = 32

    elif argv == '13':
        params['nb_cnn2d_filt'] = 64
        params['rnn_size'] = 64

    elif argv == '14':
        params['nb_cnn2d_filt'] = 128
        params['rnn_size'] = 128

    elif argv == '15':
        params['nb_cnn2d_filt'] = 256
        params['rnn_size'] = 256

    elif argv == '16':
        params['nb_cnn2d_filt'] = 512
        params['rnn_size'] = 512

    elif argv == '2':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '3':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '4':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '5':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'

    elif argv == '6':
        params['dropout_rate'] = 0.15

    elif argv == '7':
        params['dropout_rate'] = 0.3

    elif argv == '8':
        params['dropout_rate'] = 0.5

    elif argv == '9':
        params['branch_weights'] = [1, 10]

    elif argv == '10':
        params['branch_weights'] = [1, 100]

    elif argv == '11':
        params['branch_weights'] = [1, 1000]

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached
    params['unique_classes'] = 2 # maximum number of overlapping sound events
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
