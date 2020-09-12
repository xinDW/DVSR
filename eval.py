import os
import re
import time

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio 

from config import config
from utils import write3d, interpolate3d, get_file_list, load_im, _raise, exists_or_mkdir, normalize_max, normalize_percentile
from model import DBPN, res_dense_net, unet3d, unet_care, denoise_net
from model.util import save_graph_as_pb, convert_graph_to_fp16, load_graph, Model, Predictor, LargeDataPredictor


using_batch_norm = config.using_batch_norm 
normalization    = config.normalization

checkpoint_dir = config.TRAIN.ckpt_dir
pb_file_dir    = 'checkpoint/pb/'
lr_size        = config.VALID.lr_img_size 

valid_lr_img_path = config.VALID.lr_img_path
save_dir          = config.VALID.saving_path


device_id = config.TRAIN.device_id
conv_kernel = config.TRAIN.conv_kernel

label = config.label
archi1 = config.archi1
archi2 = config.archi2

factor = 1 if archi2 == 'unet' else config.factor


input_op_name     = 'Placeholder'
output_op_name  = 'net_s2/out/Tanh' # 
# output_op_name    = 'net_s2/out/Identity'

# if archi == 'unet':
#     nc = lr_size[-1]
#     lr_size = [i * 4 for i in lr_size]
#     lr_size[-1] = nc
#     lr_transform = interpolate3d
# else:
#     lr_transform = None



def build_model_and_load_npz(epoch, use_cpu=False, save_pb=False):
    
    epoch = 'best' if epoch == 0 else epoch
    # # search for ckpt files 
    def _search_for_ckpt_npz(file_dir, tags):
        filelist = os.listdir(checkpoint_dir)
        for filename in filelist:
            if '.npz' in filename:
                if all(tag in filename for tag in tags):
                    return filename
        return None

    if (archi1 is not None):
        resolve_ckpt_file = _search_for_ckpt_npz(checkpoint_dir, ['resolve', str(epoch)])
        interp_ckpt_file  = _search_for_ckpt_npz(checkpoint_dir, ['interp', str(epoch)])
       
        (resolve_ckpt_file is not None and interp_ckpt_file is not None) or _raise(Exception('checkpoint file not found'))

    else:
        #checkpoint_dir = "checkpoint/" 
        #ckpt_file = "brain_conv3_epoch1000_rdn.npz"
        ckpt_file = _search_for_ckpt_npz(checkpoint_dir, [str(epoch)])
        
        ckpt_file is not None or _raise(Exception('checkpoint file not found'))
    

    #======================================
    # build the model
    #======================================
    
    if use_cpu is False:
        device_str = '/gpu:%d' % device_id
    else:
        device_str = '/cpu:0'

    LR = tf.placeholder(tf.float32, [1] + lr_size)
    if (archi1 is not None):
        # if ('resolve_first' in archi):        
        with tf.device(device_str):
            if archi1 =='dbpn':   
                resolver = DBPN(LR, upscale=False, name="net_s1")
            elif archi1 =='denoise': 
                resolver = denoise_net(LR, name="net_s1")
            else:
                _raise(ValueError())
            
            if archi2 =='rdn':
                interpolator = res_dense_net(resolver.outputs, factor=factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, name="net_s2")
                net = interpolator
            else:
                _raise(ValueError())

    else : 
        archi = archi2
        with tf.device(device_str):
            if archi =='rdn':
                net = res_dense_net(LR, factor=factor, bn=using_batch_norm, conv_kernel=conv_kernel, name="net_s2")
            elif archi =='unet':
                # net = unet3d(LR, upscale=False)
                net = unet_care(LR)
            elif archi =='dbpn':
                net = DBPN(LR, upscale=True)
            else:
                raise Exception('unknow architecture: %s' % archi)

    net.print_params(details=False)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if (archi1 is None):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)
    else:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)

    return sess, net, LR

def save_as_pb(graph_file_tag, sess):
    tl.files.exists_or_mkdir(pb_file_dir)

    
    graph_file_16bit  = '%s_half-precision.pb' % (graph_file_tag)
    graph_file_32bit  = '%s.pb' % (graph_file_tag)
    graph_file = os.path.join(pb_file_dir, graph_file_32bit) 
    save_graph_as_pb(sess=sess, 
        output_node_names=output_op_name, 
        output_graph_file=graph_file)

    convert_graph_to_fp16(graph_file, pb_file_dir, graph_file_16bit, as_text=False, target_type='fp16', input_name=input_op_name, output_names=[output_op_name])

def get_series_percentile(filelist, path, thres=[2, 99.8]):
    print('estimating series percentile ... ')

    thres_l, thres_h = thres
    percentiles_h = []
    percentiles_l = []
    for im_file in filelist:
        im = imageio.volread(os.path.join(path, im_file))
        pl, ph = np.percentile(im, [thres_l, thres_h])
        percentiles_h.append(ph)
        percentiles_l.append(pl)
        print('%s : [%.2f, %.2f]' % (im_file, pl, ph))

    pctl_h = np.median(np.asarray(percentiles_h))
    pctl_l = np.median(np.asarray(percentiles_l))
    print('final percentiles: %.2f@%.2f\%, %.2f@%.2f\%' % (pctl_l, thres_l, pctl_h, thres_h))
    return pctl_l, pctl_h

def evaluate_whole(epoch, load_graph_from_pb=False, half_precision_infer=False, use_cpu=False, large_volume=False, save_pb=True, save_activations=False):
    
    start_time = time.time()
    
    device_tag = 'gpu' if not use_cpu else 'cpu'
    graph_file_tag = '%s_%dx%dx%d_%s' % (label.replace("/", "-"), lr_size[0], lr_size[1], lr_size[2], device_tag)

    if load_graph_from_pb:
        graph_file  = '%s_half-precision.pb' % (graph_file_tag) if half_precision_infer else graph_file_tag + '.pb'
        model_path = os.path.join(pb_file_dir, graph_file)
        os.path.exists(model_path) or _raise(ValueError('%s doesn\'t exist' % model_path))

        import_name = "hp"
        sess = load_graph(model_path, import_name=import_name, verbose=False)

        LR   = sess.graph.get_tensor_by_name("%s/%s:0" % (import_name, input_op_name))
        net  = sess.graph.get_tensor_by_name("%s/%s:0" % (import_name, output_op_name))

    else:
        
        sess, net, LR = build_model_and_load_npz(epoch, use_cpu=use_cpu, save_pb=save_pb)
        if save_pb:
            save_as_pb(graph_file_tag, sess=sess)
            return

    exists_or_mkdir(save_dir)
    model      = Model(net, sess, LR)

    block_size = lr_size[0:3]
    overlap    = 0.2
    
      
    if large_volume:
        start_time = time.time()
        predictor = LargeDataPredictor(data_path=valid_lr_img_path, 
            saving_path=save_dir, 
            factor=factor, 
            model=model, 
            block_size=block_size,
            overlap=overlap,
            half_precision=half_precision_infer)
        predictor.predict()
        print('time elapsed : %.2fs' % (time.time() - start_time))

    else:  
        valid_lr_imgs = get_file_list(path=valid_lr_img_path, regx='.*.tif') 
        predictor = Predictor(factor=factor, model=model, half_precision=half_precision_infer)

        for _, im_file in enumerate(valid_lr_imgs):
            start_time = time.time()
            
            print('='*66)
            print('predicting on %s ' % os.path.join(valid_lr_img_path, im_file) )
            
            im = imageio.volread(os.path.join(valid_lr_img_path, im_file))
            if archi1 is None and archi2 == 'unet':
                im = interpolate3d(im, factor=config.factor)

            # if (thres > 100):
            #     sr = predictor.predict(im, block_size, overlap, normalization='fixed', max_v=thres / 100.)
            # else:
            #     sr = predictor.predict(im, block_size, overlap, normalization='percentile', low=0.2, high=thres )
            
            sr = predictor.predict(im, block_size, overlap, normalization=normalization)

            print('time elapsed : %.4f' % (time.time() - start_time))
            # imageio.volwrite(os.path.join(save_dir, ('SR-thres%s-' % str(thres).replace('.', 'p')) + im_file), sr)
            try:
                imageio.volwrite(os.path.join(save_dir, 'SR-' + im_file), sr)

            except ValueError: # data too large for standard TIFF file
                short_name = im_file.split('.tif')[0].replace('.', '_')
                slice_save_dir   = os.path.join(save_dir, 'SR-' + short_name)
                exists_or_mkdir(slice_save_dir)

                for d, layer in enumerate(sr):
                    name = os.path.join(slice_save_dir, '%05d.tif' % (d + 1) )
                    imageio.imwrite(name, layer) 
            
                
    model.recycle()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=int, default=0)
    # parser.add_argument("-r", "--rdn", help="use one-stage(RDN) net for inference",
    #                     action="store_true") #if the option is specified, assign True to args.rdn. Otherwise False.
    parser.add_argument("--pb", help="load graph from pb file instead of buiding from API",
                        action="store_true") 

    parser.add_argument("--cpu", help="use cpu for inference",
                        action="store_true") 

    parser.add_argument("-f", "--half_precision", help="use half-precision model for inference",
                        action="store_true") 

    parser.add_argument("--large", help="predict on large volume", 
                        action="store_true")

    parser.add_argument("-p", "--save_pb", help="save the loaded graph as a half-precision pb file",
                        action="store_true") 

    parser.add_argument("-l", "--layer", help="save activations of each layer",
                        action="store_true") 

    # parser.add_argument("--stage1", help="run stage1 only",
    #                     action="store_true") 
    # parser.add_argument("--stage2", help="run stage2 only",
    #                     action="store_true") 

    args = parser.parse_args()

    evaluate_whole(epoch=args.ckpt, load_graph_from_pb=args.pb, half_precision_infer=args.half_precision, use_cpu=args.cpu, large_volume=args.large, save_pb=args.save_pb, save_activations=args.layer)
