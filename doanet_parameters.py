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
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='/scratch/asignal/sharath/DCASE2020_SELD_dataset/',

        # OUTPUT PATH
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
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
        dropout_rate=0.,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=128,           # Number of CNN nodes, constant for each layer
        f_pool_size=[2, 2, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,        # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=2,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_fnn_act_layers=2,
        fnn_act_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=200,              # Train for maximum epochs
        lr=1e-3,
        dMOTA_wt = 1,
        dMOTP_wt = 50,
        IDS_wt = 1,
        branch_weights=[1, 10.],
        use_dmot_only=False,
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        params['dataset'] = 'foa'
        params['use_hnet']= False
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_baseline/'
    elif argv == '3':
        params['dataset'] = 'foa'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label/'

    elif argv == '4':
        params['dataset'] = 'foa'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label/'

    elif argv == '5':
        params['dataset'] = 'foa'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'

    elif argv == '6':
        params['dataset'] = 'foa'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'

    elif argv == '7':
        params['dataset'] = 'mic'
        params['use_hnet']= False
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_baseline/'
    elif argv == '8':
        params['dataset'] = 'mic'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label/'

    elif argv == '9':
        params['dataset'] = 'mic'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label/'

    elif argv == '10':
        params['dataset'] = 'mic'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'

    elif argv == '11':
        params['dataset'] = 'mic'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'

    elif argv == '12':
        params['dataset'] = 'foa'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 0
        params['IDS_wt']= 0

    elif argv == '13':
        params['dataset'] = 'foa'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 0

    elif argv == '14':
        params['dataset'] = 'foa'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 1

    elif argv == '15':
        params['dataset'] = 'foa'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 0
        params['IDS_wt']= 0

    elif argv == '16':
        params['dataset'] = 'foa'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 0

    elif argv == '17':
        params['dataset'] = 'foa'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 1

    elif argv == '18':
        params['dataset'] = 'mic'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 0
        params['IDS_wt']= 0

    elif argv == '19':
        params['dataset'] = 'mic'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 0

    elif argv == '20':
        params['dataset'] = 'mic'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= True
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 1

    elif argv == '21':
        params['dataset'] = 'mic'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 0
        params['IDS_wt']= 0

    elif argv == '22':
        params['dataset'] = 'mic'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 0

    elif argv == '23':
        params['dataset'] = 'mic'
        params['feat_label_dir']='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label_augmented/'
        params['use_hnet']= True
        params['use_dmot_only']= False
        params['dMOTP_wt']= 50
        params['dMOTA_wt']= 1
        params['IDS_wt']= 1

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
