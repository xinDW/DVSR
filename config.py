from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.VALID = edict()


config.TRAIN.img_size_lr = [20, 20, 20, 1] #[d,h,w,c]
config.TRAIN.img_size_hr = [80, 80, 80, 1]

## if use multi-gpu training
config.TRAIN.num_gpus = 2
## if use single-gpu training  
config.TRAIN.device_id = 1

config.TRAIN.using_mixed_precision = False
config.TRAIN.using_batch_norm  = False
config.TRAIN.using_edge_loss = True
#label = '3T3_488_bg_factor4_multigpu_cross_entropy'
#label = '3T3_488_sbg_2stages_factor4_mse'
#label = 'celegans_simu_by_zhaoyuxuan_2stage_dbpn+rdn_factor4_mse'
#label = 'whole_brain_training_hr_step1um_2stage_dbpn+rdn_factor4_mse'
#label = 'brain_simu_by_fangchunyu_hr_step1um_lr_20fps_training_data_xz-transposed_2stage_dbpn+rdn_factor4_mse'
#label = 'whole+half_brain_training_hr_raw_step1um_2stage_dbpn+rdn_factor4_mse'
label = 'whole+half_brain_training_hr_raw_step1um_2stage_dbpn+rdn_factor4_mse+edge'
#label = 'whole+half_brain_training_hr_raw_step1um_2stage_dbpn+rdn_interp_first_factor4_mse'
#label =  'test_mixed_precision'
config.label = label
#config.archi = '2stage_interp_first'
config.archi = '2stage_resolve_first'

config.TRAIN.lr_img_path = "data/brain/brain20190316/crop80X80X80/test/lr/"
config.TRAIN.hr_img_path = "data/brain/brain20190316/crop80X80X80/test/hr/"
config.TRAIN.mr_img_path = "data/brain/brain20190316/crop80X80X80/test/mr/"
config.TRAIN.test_data_path = "data/brain/brain20190316/crop80X80X80/test/"
#config.TRAIN.lr_img_path = "data/train/3T3/488/factor4_sbg/cropped64X64X16_overlap0.20-0.20-0.20/all/dynamic_range_adjusted/"
#config.TRAIN.hr_img_path = "data/train/3T3/488/factor4_sbg/cropped256X256X64_overlap0.20-0.20-0.20/all/dynamic_range_adjusted/"
#config.TRAIN.mr_img_path = "data/train/3T3/488/factor4_sbg/cropped256X256X64_overlap0.20-0.20-0.20/all/dynamic_range_adjusted/ds/"
#config.TRAIN.lr_img_path = "data/brain/brain_simu_by_fangchunyu/training_HR_step1um_LR_step4um/cropped40X40X20_overlap0.20-0.20-0.20/all/"
#config.TRAIN.hr_img_path = "data/brain/brain_simu_by_fangchunyu/training_HR_step1um_LR_step4um/cropped160X160X80_overlap0.20-0.20-0.20/all/"
#config.TRAIN.mr_img_path = "data/brain/brain_simu_by_fangchunyu/training_HR_step1um_LR_step4um/cropped160X160X80_overlap0.20-0.20-0.20/ds/all/"

config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(label)
config.TRAIN.log_dir = "log/{}/".format(label)
config.TRAIN.batch_size = 1
config.TRAIN.beta1 = 0.9
config.TRAIN.n_epoch = 4000
config.TRAIN.decay_every = 10
config.TRAIN.lr_decay = 0.5

config.TRAIN.conv_kernel = 3
config.TRAIN.learning_rate_init = 1e-4

config.VALID.lr_img_path = "data/brain/DVSR data/"
config.VALID.lr_img_size = [20,40,40,1] # [depth, height, width, channels]
#config.VALID.saving_path = "sample/validate/conv_kernel%d/" % config.TRAIN.conv_kernel
config.VALID.saving_path = "data/recon2/{}/".format(label)
config.VALID.on_the_fly = False