from easydict import EasyDict as edict
from utils import parse_label


train_img_size_lr = [32, 32, 32, 1] #[d,h,w,c]

label = 'tubulin3d-simu_2stage-dbpn+rdn_factor-4_norm-fixed_loss-mse'  

train_lr_img_path = "data/example-data/tubulin/train/lr/"
train_hr_img_path = "data/example-data/tubulin/train/hr/"
train_mr_img_path = "data/example-data/tubulin/train/mr/"

using_batch_norm = False
train_test_data_path = None
train_valid_lr_path  = None    # valid on_the_fly 

valid_lr_img_path = "data/example-data/tubulin/inference/lr/"

valid_block_size = [64,64,64,1] # [depth, height, width, channels]
valid_block_overlap = 0.2





config = edict()
config.TRAIN = edict()
config.VALID = edict()

params = parse_label(label)
archi2 = params['archi2']  #['rdn', 'unet', 'dbpn']
archi1 = params['archi1']  # [None, 'dbpn' 'denoise'] # None if 1stage
loss   = params['loss']    #['mse', 'mae']
factor = params['factor']
normalization = params['norm']

config.archi1 = archi1
config.archi2 = archi2
config.loss   = loss

config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.batch_size = 1
config.TRAIN.beta1 = 0.9
config.TRAIN.n_epoch = 500
config.TRAIN.decay_every = 50
config.TRAIN.lr_decay = 0.5
config.TRAIN.conv_kernel = 3
config.TRAIN.learning_rate_init = 1e-4


config.factor = factor
config.TRAIN.img_size_lr = train_img_size_lr
config.TRAIN.img_size_hr = [train_img_size_lr[0]*factor, train_img_size_lr[1]*factor, train_img_size_lr[2]*factor, 1]
config.using_batch_norm  = using_batch_norm
config.normalization     = normalization
config.label = label


## if use multi-gpu training
config.TRAIN.num_gpus = 1
## if use single-gpu training  
config.TRAIN.device_id = 2

config.TRAIN.using_mixed_precision = False

config.TRAIN.using_edge_loss = False
config.TRAIN.using_grad_loss = False

config.TRAIN.lr_img_path = train_lr_img_path
config.TRAIN.hr_img_path = train_hr_img_path
config.TRAIN.mr_img_path = train_mr_img_path

config.TRAIN.test_data_path = train_test_data_path
config.TRAIN.valid_lr_path = train_valid_lr_path  # valid on_the_fly 

config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_dir         = "checkpoint/{}/".format(label)
config.TRAIN.log_dir          = "log/{}/".format(label)

config.VALID.lr_img_path   = valid_lr_img_path
config.VALID.block_size    = valid_block_size # [depth, height, width, channels]
config.VALID.block_overlap = valid_block_overlap
config.VALID.saving_path   = "{}/{}/".format(config.VALID.lr_img_path, label)