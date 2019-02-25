from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.TRAIN_DFE = edict()
config.VALID = edict()


config.TRAIN.img_size_lr = [8, 64, 64, 1] #[d,h,w,c]
config.TRAIN.img_size_hr = [32, 256, 256, 1]

## if use multi-gpu training
config.TRAIN.num_gpus = 2
## if use single-gpu training 
config.TRAIN.device_id = 0

config.TRAIN.using_edges_loss = False

#label = '3T3_488_bg_factor4_multigpu_cross_entropy'
label = '3T3_488_bg_2stages_factor4_mse'

#if config.TRAIN.using_edges_loss:
  #abel = label + '+edges' 

#config.TRAIN.lr_img_path = "data/3T3/train/factor4_bg/cropped64X64X8_overlap0.50-0.50-0.50-488/all/"
#config.TRAIN.hr_img_path = "data/3T3/train/factor4_bg/cropped256X256X32_overlap0.50-0.50-0.50-488/all/"
#config.TRAIN.mr_img_path = "data/3T3/train/factor4_bg/cropped256X256X32_overlap0.50-0.50-0.50-488/all/ds/"
config.TRAIN.lr_img_path = "data/train/3T3/488/factor4_bg/cropped64X64X8_overlap0.50-0.50-0.50/all/"
config.TRAIN.hr_img_path = "data/train/3T3/488/factor4_bg/cropped256X256X32_overlap0.50-0.50-0.50/all/"
config.TRAIN.mr_img_path = "data/train/3T3/488/factor4_bg/cropped256X256X32_overlap0.50-0.50-0.50//all/ds/"
#config.TRAIN.lr_img_path = "data/brain/Registration_for_training/cropped64X64X16_overlap0.50-0.50-0.50/"
#config.TRAIN.hr_img_path = "data/brain/Registration_for_training/cropped256X256X64_overlap0.50-0.50-0.50/"
config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(label)
config.TRAIN.log_dir = "log/{}/".format(label)
config.TRAIN.batch_size = 1
config.TRAIN.beta1 = 0.9
config.TRAIN.n_epoch = 4000
config.TRAIN.decay_every = 200
config.TRAIN.lr_decay = 0.5

config.TRAIN.conv_kernel = 3
config.TRAIN.learning_rate_init = 1e-4

config.VALID.lr_img_path = "data/test/"
#config.VALID.lr_img_path = "data/validate/zebrafish_heart/"
config.VALID.lr_img_size = [20,100,100,1]
#config.VALID.saving_path = "sample/validate/conv_kernel%d/" % config.TRAIN.conv_kernel
config.VALID.saving_path = "data/test/3T3/recon2/"
config.VALID.on_the_fly = False