import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time

from config import config
from utils import read_all_images, write3d
from model import DBPN, res_dense_net

def evaluate(epoch, archi='2stage'):

    using_batch_norm = config.TRAIN.using_batch_norm 

    
    checkpoint_dir = config.TRAIN.ckpt_dir
    lr_size = config.VALID.lr_img_size
    valid_lr_img_path = config.VALID.lr_img_path
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)
    
    start_time = time.time()
    valid_lr_imgs = read_all_images(valid_lr_img_path, lr_size[0])
    if archi == '2stage':
        archi = config.archi
    else:
        archi = 'RDN'
 
    device_id = config.TRAIN.device_id
    conv_kernel = config.TRAIN.conv_kernel
    using_batch_norm = config.TRAIN.using_batch_norm 

    resolve_ckpt_found = False
    interp_ckpt_found = False

    #======================================
    # search for ckpt files 
    #======================================
    if ('2stage' in archi):
        label = config.label
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
    else:
        label = 'RDN'
        checkpoint_dir = "checkpoint/" 
        ckpt_file = "brain_conv3_epoch1000_rdn.npz"
    

    #======================================
    # build the model
    #======================================
    LR = tf.placeholder(tf.float32, [1] + lr_size)
    if ('2stage' in archi):
            if ('resolve_first' in archi):
                with tf.device('/gpu:%d' % device_id):
                    resolver = DBPN(LR, upscale=False, name="net_s1")
                #with tf.device('/gpu:1'):
                    interpolator = res_dense_net(resolver.outputs, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, name="net_s2")
                    net = interpolator
            else :
                with tf.device('/gpu:0'):
                    interpolator = res_dense_net(LR, factor=4, conv_kernel=conv_kernel, reuse=False, bn=using_batch_norm, is_train=True, name="net_s1")
                with tf.device('/gpu:1'):
                    resolver = DBPN(interpolator.outputs, upscale=False, name="net_s2")
                    net = resolver

    else : 
        with tf.device('/gpu:1'):
            net = res_dense_net(LR, reuse=False, name="rdn")

    
    #======================================
    # assign trained parameters
    #======================================        
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        if ('2stage' in archi):
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)
        else:
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)

        #======================================
        # begin to inference
        #======================================
        for idx in range(0,len(valid_lr_imgs)):
            
            SR = sess.run(net.outputs, {LR : valid_lr_imgs[idx:idx+1]})
            #MR = sess.run(resolver.outputs, {LR : valid_lr_imgs[idx:idx+1]})
            write3d(SR, save_dir+'SR_%s_%06d_epoch%d.tif' % (label, idx, epoch))
            #write3d(MR, save_dir+'MR_%s_%06d_epoch%d.tif' % (label, idx, epoch))
            print('writing %d / %d ...' % (idx + 1, len(valid_lr_imgs)))

    print("time elapsed : %4.4fs " % (time.time() - start_time))
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--archi', default="2stage")
    args = parser.parse_args()
    ckpt = args.ckpt
    archi = args.archi

    evaluate(ckpt, archi)
    
