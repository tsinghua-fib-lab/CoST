import os
from easydict import EasyDict as edict


def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


def default_config(data='SST'):
    ws =  get_workspace()
    config = edict()
    config.PATH_MOD = ws + '/save/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/dataset/data/'


    if config.data.name == 'CrowdBJ':
        config.data.num_features = 1
        config.data.num_vertices = 1010
        config.data.points_per_hour = 1
        config.data.path= config.data.path + 'Urban/CrowdBJ.npy'

    
    if config.data.name == 'CrowdBM':
        config.data.num_features = 1
        config.data.num_vertices = 403
        config.data.points_per_hour = 1
        config.data.path= config.data.path + 'Urban/CrowdBM.npy'

    if config.data.name == 'MobileNJ':
        config.data.num_features = 1
        config.data.num_vertices = 560
        config.data.points_per_hour = 1
        config.data.path= config.data.path + 'CommTraffic/MobileNJ.npy'


    if config.data.name == 'MobileSH':
        config.data.num_features = 1
        config.data.num_vertices = 896
        config.data.points_per_hour = 1
        config.data.path= config.data.path + 'CommTraffic/MobileSH.npy'

    
    if config.data.name == 'BikeDC':
        config.data.num_features = 1
        config.data.num_vertices = 400
        config.data.points_per_hour = 2
        config.data.path= config.data.path + 'Urban/BikeDC.npy'


    if config.data.name == 'TaxiBJ':
        config.data.num_features = 1
        config.data.num_vertices = 1024
        config.data.points_per_hour = 2
        config.data.path= config.data.path + 'Urban/TaxiBJ.npy'
    

    if config.data.name == 'Los_Speed':
        config.data.num_features = 1
        config.data.num_vertices = 207
        config.data.points_per_hour = 12
        config.data.path= config.data.path + 'Urban/Los_Speed.npy'


    if config.data.name == 'SST':
        config.data.num_features = 1
        config.data.num_vertices = 500
        config.data.points_per_year = 12
        config.data.path= config.data.path + 'Climate/SST'




    # model config
    config.model_det = edict()
    config.model_det.node_dim = 32 if config.data.name != "SST" else 64
    config.model_det.input_dim = 1
    config.model_det.embed_dim = 32 if config.data.name != "SST" else 64
    config.model_det.num_layer = 4 if config.data.name != "SST" else 8
    config.model_det.temp_dim_tid = 32 if config.data.name != "SST" else 64
    config.model_det.temp_dim_diw = 32 
    config.model_det.if_time_in_day = True if config.data.name != "SST" else False
    config.model_det.time_of_day_size = int(24/config.data.points_per_hour) if config.data.name != "SST" else 12
    config.model_det.if_day_of_week = True 
    config.model_det.day_of_week_size = 7 if config.data.name != "SST" else 12 
    config.model_det.if_spatial = True

    config.model_diff = edict()
    config.model_diff.node_dim = 32 
    config.model_diff.input_dim = 2
    config.model_diff.embed_dim = 128
    config.model_diff.num_layer = 8
    config.model_diff.temp_dim_tid = 32 
    config.model_diff.temp_dim_diw = 32 
    config.model_diff.if_time_in_day = True if config.data.name != "SST" else False
    config.model_diff.time_of_day_size = int(24/config.data.points_per_hour) if config.data.name != "SST" else 12
    config.model_diff.if_day_of_week = True 
    config.model_diff.day_of_week_size = 7 if config.data.name != "SST" else 12 
    config.model_diff.if_spatial = True
    config.model_diff.n_samples = 50
    config.model_diff.n_samples_val = 3
    config.model_diff.diff_steps = 50



    # training config
    config.batch_size = 8
    config.weight_decay=1e-5
    config.lr = 1e-3


    return config


def get_config(args):
    config = default_config(args.data_name)
    config.model_det.input_dim = args.channels
    config.model_diff.input_dim = args.channels*2
    config.model_diff.n_samples = args.nsample
    config.model_diff.n_samples_val = 3
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.patience = args.patience
    config.val_batch_size = args.val_batch_size
    config.device = args.device
    config.seed = args.seed
    config.unconditional = args.unconditional
    config.epochs = args.epochs
    config.device = args.device
    config.eps_model = args.eps_model
    config.history_len = args.history_len
    config.predict_len = args.predict_len
    # config.mean_model = args.mean_model

    return config