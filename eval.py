import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time

from config import *
from utils import read_all_images, write3d
from model import DBPN, res_dense_net

def evaluate(epoch):

    using_batch_norm = config.TRAIN.using_batch_norm 

    #checkpoint_dir = "checkpoint/"
    checkpoint_dir = config.TRAIN.ckpt_dir
    lr_size = config.VALID.lr_img_size
    valid_lr_img_path = config.VALID.lr_img_path
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)
    
    start_time = time.time()
    valid_lr_imgs = read_all_images(valid_lr_img_path, lr_size[0])
    
    LR = tf.placeholder("float", [1] + lr_size)
    with tf.device('/gpu:0'):
        resolver = DBPN(LR, upscale=False, name='resolver')
    with tf.device('/gpu:1'):
        interpolator = res_dense_net(resolver.outputs, reuse=False, bn=using_batch_norm,  name='interpolator')

  
    resolve_ckpt_found = False
    interp_ckpt_found = False

    filelist = os.listdir(checkpoint_dir)
    for file in filelist:
        if '.npz' in file and str(epoch) in file:
            if 'resolve' in file:
                resolve_ckpt_file = file
                resolve_ckpt_found = True
            if 'interp' in file:
                interp_ckpt_file = file
                interp_ckpt_found = True
            if resolve_ckpt_found and interp_ckpt_found:
                break

    #interp_ckpt_file = 'checkpoint/brain_conv3_epoch1000.npz'

    if not (resolve_ckpt_found and interp_ckpt_found):
        raise Exception('no such checkpoint file')
            
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)
        #tl.files.load_and_assign_npz(sess=sess, name=interp_ckpt_file, network=interpolator)
        
        for idx in range(0,len(valid_lr_imgs)):
            
            MR, SR = sess.run([resolver.outputs, interpolator.outputs], {LR : valid_lr_imgs[idx:idx+1]})
            #MR = sess.run(resolver.outputs, {LR : valid_lr_imgs[idx:idx+1]})
            write3d(SR, save_dir+'SR_%s_%06d_epoch%d.tif' % (label, idx, epoch))
            #write3d(MR, save_dir+'MR_%s_%06d_epoch%d.tif' % (label, idx, epoch))
            print('writing %d / %d ...' % (idx + 1, len(valid_lr_imgs)))
    print("time elapsed : %4.4fs " % (time.time() - start_time))
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=0)
    args = parser.parse_args()
    ckpt = args.ckpt

    evaluate(ckpt)
    
