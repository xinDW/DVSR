import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import re
import time

from config import config
from utils import read_all_images, write3d, interpolate3d, get_file_list, load_im, _raise
from model import DBPN, res_dense_net, unet3d, denoise_net
from model.util import save_graph_as_pb, convert_graph_to_fp16, load_graph, Model, Predictor, LargeDataPredictor


using_batch_norm = config.using_batch_norm 

checkpoint_dir = config.TRAIN.ckpt_dir
pb_file_dir    = 'checkpoint/pb/'
lr_size        = config.VALID.lr_img_size 

valid_lr_img_path = config.VALID.lr_img_path
save_dir = config.VALID.saving_path
tl.files.exists_or_mkdir(save_dir)

device_id = config.TRAIN.device_id
conv_kernel = config.TRAIN.conv_kernel

label = config.label
archi1 = config.archi1
archi2 = config.archi2

factor = 1 if archi2 is 'unet' else config.factor

graph_file_16bit   = '%s_%dx%dx%d_half-precision.pb' % (label.replace("/", "-"), lr_size[0], lr_size[1], lr_size[2])
input_op_name  = 'Placeholder'
# output_op_name = 'net_s2/out/Tanh' # 
output_op_name = 'net_s2/out/Identity'

# if archi is 'unet':
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
            if archi1 is 'dbpn':   
                resolver = DBPN(LR, upscale=False, name="net_s1")
            elif archi1 is 'denoise': 
                resolver = denoise_net(LR, name="net_s1")
            else:
                _raise(ValueError())
            
            if archi2 is 'rdn':
                interpolator = res_dense_net(resolver.outputs, factor=factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, name="net_s2")
                net = interpolator
            else:
                _raise(ValueError())

    else : 
        archi = archi2
        with tf.device(device_str):
            if archi is 'rdn':
                net = res_dense_net(LR, factor=factor, bn=using_batch_norm, conv_kernel=conv_kernel)
            elif archi is 'unet':
                net = unet3d(LR, upscale=False, is_train=False)
            elif archi is 'dbpn':
                net = DBPN(LR, upscale=True)
            else:
                raise Exception('unknow architecture: %s' % archi)

    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if (archi1 is None):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)
    else:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)

    if save_pb:
        tl.files.exists_or_mkdir(pb_file_dir)

        graph_file = os.path.join(save_dir, 'best-model.pb') 
        save_graph_as_pb(sess=sess, 
            output_node_names=output_op_name, 
            output_graph_file=graph_file)

        convert_graph_to_fp16(graph_file, pb_file_dir, graph_file_16bit, as_text=False, target_type='fp16', input_name=input_op_name, output_names=[output_op_name])

    return sess, net, LR

def evaluate_whole(epoch, half_precision_infer=False, use_cpu=False, large_volume=False, save_pb=True, save_activations=False):
    
    start_time = time.time()

    valid_lr_imgs = get_file_list(path=valid_lr_img_path, regx='.*.tif')

    if half_precision_infer:
        model_path = os.path.join(pb_file_dir, graph_file_16bit)
        os.path.exists(model_path) or _raise(ValueError('%s doesn\'t exist' % model_path))

        import_name = "hp"
        sess = load_graph(model_path, import_name=import_name)

        for op in sess.graph.get_operations():
            print(op.name)

        LR   = sess.graph.get_tensor_by_name("%s/%s:0" % (import_name, input_op_name))
        net  = sess.graph.get_tensor_by_name("%s/%s:0" % (import_name, output_op_name))

    else:
        sess, net, LR = build_model_and_load_npz(epoch, use_cpu=use_cpu, save_pb=save_pb)
        if save_pb:
            return

    model      = Model(net, sess, LR)

    block_size = lr_size[0:3]
    overlap    = 0.2
    norm_thres = [2]

    import imageio   
    dtype = np.float16 if half_precision_infer else np.float32
    if large_volume:
        start_time = time.time()
        predictor = LargeDataPredictor(data_path=valid_lr_img_path, 
            saving_path=save_dir, 
            factor=factor, 
            model=model, 
            block_size=block_size,
            overlap=overlap,
            dtype=dtype)
        predictor.predict()
        print('time elapsed : %.2fs' % (time.time() - start_time))

    else:   
        predictor = Predictor(factor = factor, model=model, dtype=dtype)

        for _, im_file in enumerate(valid_lr_imgs):
            for _, low in enumerate(norm_thres):
                start_time = time.time()
                im = imageio.volread(os.path.join(valid_lr_img_path, im_file))
                
                print('='*66)
                print('predicting on %s ' % os.path.join(valid_lr_img_path, im_file) )
                sr = predictor.predict(im, block_size, overlap, low=low)
                print('time elapsed : %.4f' % (time.time() - start_time))
                imageio.volwrite(os.path.join(save_dir, ('SR-%d-' % low) + im_file), sr)

    model.recycle()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=int, default=0)
    # parser.add_argument("-r", "--rdn", help="use one-stage(RDN) net for inference",
    #                     action="store_true") #if the option is specified, assign True to args.rdn. Otherwise False.
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

    evaluate_whole(epoch=args.ckpt, half_precision_infer=args.half_precision, use_cpu=args.cpu, large_volume=args.large, save_pb=args.save_pb, save_activations=args.layer)

'''
def evaluate_fragments(epoch, use_cpu=False, save_activations=False, stage1_only=False, stage2_only=False):
    prefix = 'MR' if stage1_only else 'SR' 
    scale_pixel_value = False #if (stage1_only or stage2_only) else True

    start_time = time.time()

    valid_lr_imgs = read_all_images(valid_lr_img_path, lr_size[0], transform=None)

    def visualize_layers(sess, final_layer, feed_dict):
        """
        save all the feature maps (before the final_layer) into tif file.
        Params:
            -final_layer: a tl.layers.Layer instance
        """
        
        fetches = {}
        for i, layer in enumerate(final_layer.all_layers):
            print("  layer {:2}: {:40}  {:20}".format(i, str(layer.name), str(layer.get_shape())))
            name = re.sub(':', '', str(layer.name))
            name = re.sub('/', '_', name)
            fetches.update({name : layer})
        features = sess.run(fetches, feed_dict)

        layer_idx = 0
        
        for name, feat in features.items():
            save_path = save_dir + 'layers/%03d/' % layer_idx
            tl.files.exists_or_mkdir(save_path)
            filename = save_path +'{}.tif'.format(name)
            write3d(feat, filename)
            layer_idx += 1

            
    #======================================
    # search for ckpt files 
    #======================================
    if ('2stage' in archi):
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
            raise Exception('checkpoint file not found')
    else:
        #checkpoint_dir = "checkpoint/" 
        #ckpt_file = "brain_conv3_epoch1000_rdn.npz"
        ckpt_found = False
        filelist = os.listdir(checkpoint_dir)
        for file in filelist:
            if '.npz' in file and str(epoch) in file:
                ckpt_file = file
                ckpt_found = True
                break
        if not (ckpt_found):
            raise Exception('checkpoint file not found')
    

    #======================================
    # build the model
    #======================================
    
    if use_cpu is False:
        device_str = '/gpu:%d' % device_id
    else:
        device_str = '/cpu:0'

    LR = tf.placeholder(tf.float32, [1] + lr_size)
    if ('2stage' in archi):
        if ('resolve_first' in archi): 
                   
            with tf.device(device_str):
                if stage1_only:   
                    resolver = DBPN(LR, upscale=False, name="net_s1")
                    net = resolver
                elif stage2_only:
                    interpolator = res_dense_net(LR, factor=factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, name="net_s2")
                    net = interpolator
                else:
                    resolver = DBPN(LR, upscale=False, name="net_s1")
                    interpolator = res_dense_net(resolver.outputs, factor=factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, name="net_s2")
                    net = interpolator
        else :
            with tf.device('/gpu:0'):
                interpolator = res_dense_net(LR, factor=factor, conv_kernel=conv_kernel, reuse=False, bn=using_batch_norm, is_train=True, name="net_s1")
            with tf.device('/gpu:1'):
                resolver = DBPN(interpolator.outputs, upscale=False, name="net_s2")
                net = resolver

    else : 
        with tf.device(device_str):
            if archi is 'rdn':
                net = res_dense_net(LR, factor=factor, bn=using_batch_norm, conv_kernel=conv_kernel)
            elif archi is 'unet':
                net = unet3d(LR, upscale=False, is_train=False)
            elif archi is 'dbpn':
                net = DBPN(LR, upscale=True)
            else:
                raise Exception('unknow architecture: %s' % archi)

    
    #======================================
    # assign trained parameters
    #======================================        
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        if ('2stage' in archi):
            if not stage2_only:
                tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
            if not stage1_only:
                tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)
        else:
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)

        #======================================
        # begin to inference
        #======================================
        global_min = 1e10
        global_max = -1e10
        for idx in range(0,len(valid_lr_imgs)):
            if idx == 0 and save_activations:
                visualize_layers(sess, net, {LR : valid_lr_imgs[0:1]})

            SR = sess.run(net.outputs, {LR : valid_lr_imgs[idx:idx+1]})
            block_max, block_min = write3d(SR, save_dir+'%s_%06d_epoch%d.tif' % (prefix, idx, epoch), scale_pixel_value=scale_pixel_value, savemat=False)
            global_min = global_min if global_min < block_min else block_min
            global_max = global_max if global_max > block_max else block_max
            
            # if save_mr:
            #     MR = sess.run(resolver.outputs, {LR : valid_lr_imgs[idx:idx+1]})
            #     write3d(MR, save_dir+'MR__epoch%d_%06d.tif' % (epoch, idx))

            
            print('\rwriting % 5d / %d ...' % (idx + 1, len(valid_lr_imgs)), end='')

    print("\ntime elapsed : %4.4fs " % (time.time() - start_time))
    print("global range : %6.6f  %6.6f" % (global_min, global_max))

'''