import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio 

from losses import edges_loss, l1_loss, l2_loss, cross_entropy
from res_dense_net import res_dense_net
from unet3d import unet3d
from utils import *
from config import *

batch_size = config.TRAIN.batch_size
lr_size = config.TRAIN.img_size_lr # [depth height width channels]
hr_size = config.TRAIN.img_size_hr

beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch

using_edges_loss = config.TRAIN.using_edges_loss

learning_rate_init = config.TRAIN.learning_rate_init
decay_every = config.TRAIN.decay_every
lr_decay = config.TRAIN.lr_decay

checkpoint_dir = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval

test_saving_dir = config.TRAIN.test_saving_path
valid_saving_dir = config.VALID.saving_path

train_lr_img_path = config.TRAIN.lr_img_path
train_hr_img_path = config.TRAIN.hr_img_path
valid_lr_img_path = config.VALID.lr_img_path
factor = hr_size[1] / lr_size[1]

num_gpus = config.TRAIN.num_gpus

def write2file(list, filename):
    with open(filename, 'w') as file:
        for g, v in list:
            file.write('grad:{}\n vars:{}\n'.format(g, v))

def tower_loss(scope, fn, image, reference):
    '''
    calculate the total loss on a sigle tower
    Params:
        -scope: perfix identifying the tower
        -fn: loss function 
        -image: [batch, depth, height, width, channels=1]
        -reference: ground truth with the same shape as the image
    '''
    loss = fn(reference, image)
    tag = 'edges_loss' if fn == edges_loss else 'ln_loss'
    tf.add_to_collection(tag, loss)
    losses = tf.get_collection(tag, scope)
    total_loss = tf.add_n(losses, name='total_loss')

    #for l in losses + [total_loss]:
        #tf.summary.scalar(l.op.name, l)
        
    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads 
    
def train(begin_epoch):

    def __find_available_ckpt(end):
        begin = end
        while load_and_assign_ckpt(sess, checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin), generator) is False:
            begin -= 10
            if begin < 0:
                return 0
                
        print('\n\ninit ckpt found at epoch %d\n\n' % begin)    
        return begin
        
    def eval_on_the_fly(epoch, init_training=True):
        if config.VALID.on_the_fly:
            out_eval = sess.run(generator_test.outputs, {LR : valid_lr})
            if init_training:
                saving_path = valid_saving_dir+'eval_epoch{}_init.tif'.format(epoch)
            else:
                saving_path = valid_saving_dir+'eval_epoch{}.tif'.format(epoch)
                
            write3d(out_eval, saving_path)   

    
    with tf.device('/cpu:0'):
        with tf.variable_scope('learning_rate'):
            learning_rate_var = tf.Variable(learning_rate_init, trainable=False)
            
        optimizer = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1)    
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate_var)  
    
        LR_for_all_tower = tf.placeholder("float", [batch_size * num_gpus] + lr_size)
        HR_for_all_tower = tf.placeholder("float", [batch_size * num_gpus] + hr_size)
    
        LR = tf.split(LR_for_all_tower, num_gpus, axis=0)
        HR = tf.split(HR_for_all_tower, num_gpus, axis=0)

        tower_grads = []
        for id in range(num_gpus):
            with tf.variable_scope('generator', reuse=id > 0):
                with tf.device('/gpu:{}'.format(id)):
                    with tf.name_scope('tower_%d' % id) as scope:  
                        generator = res_dense_net(LR[id], reuse=id>0)
                        generator.print_params(False)
                        ln_loss = tower_loss(scope, l2_loss, generator.outputs, HR[id])
                        #ln_loss = l1_loss(generator.outputs, HR[id])
                        e_loss = 0
                        if using_edges_loss:
                            #e_loss =  1e-9 * edges_loss(generator.outputs, HR[id]) 
                            e_loss = 1e-11 * tower_loss(scope, cross_entropy, generator.outputs, HR[id])

                        t_loss = ln_loss + e_loss
                        
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        
                        g_vars = tl.layers.get_variables_with_name('RDN')
                        grads_and_vars = optimizer.compute_gradients(t_loss, var_list=g_vars)
                        #grads_and_vars = optimizer.compute_gradients(t_loss)
                        tower_grads.append(grads_and_vars)

        avg_grads_and_vars = average_gradients(tower_grads)
        update_params_op = optimizer.apply_gradients(avg_grads_and_vars)

        summaries.append(tf.summary.scalar('learning_rate', learning_rate_var))
        #tf.summary.scalar('learning_rate', learning_rate_var)
        ## add histograms for gradients
        for grad, var in avg_grads_and_vars:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                tf.summary.histogram(var.op.name + '/gradients', grad)
        summary_op = tf.summary.merge(summaries)
        #summary_op = tf.summary.merge_all()
        
        '''
        # for test
        LR_test = tf.placeholder("float", [batch_size] + lr_size)
        with tf.device('/gpu:3'):
          with tf.variable_scope('generator', reuse=True):
            generator_test = res_dense_net(LR_test, reuse=True)
        '''
        
    def __train(begin_epoch, n_epoch, LR_t, HR_t, test_lr):
        summary_writer = tf.summary.FileWriter('./log', tf.get_default_graph())
        print('pre-training')
        #available_img_nums = len(HR_t) - len(HR_t)%(batch_size * num_gpus)
        for epoch in range(begin_epoch, n_epoch+1):
            
            # update learning rate
            if epoch != begin_epoch and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(learning_rate_var, learning_rate_init * new_lr_decay))
                print('[!] learning rate : %f' % (learning_rate_init * new_lr_decay))

            total_loss, total_ln_loss, total_e_loss, n_iter = 0, 0, 0, 0
            epoch_time = time.time()
            for idx in range(batch_size*num_gpus, len(HR_t), batch_size * num_gpus):
                step_time = time.time()
                
                if idx + batch_size * num_gpus >= len(HR_t):
                    continue
                    
                HR_batch = HR_t[idx : idx + batch_size * num_gpus]
                LR_batch = LR_t[idx : idx + batch_size * num_gpus]
                
                if using_edges_loss:
                    error_ln, error_e,  _ = sess.run([ln_loss, e_loss, update_params_op], {LR_for_all_tower : LR_batch, HR_for_all_tower : HR_batch})
                    print("Epoch:[%d/%d]iter:[%d/%d] times: %4.3fs, mse:%.6f, edges:%.9f" % (epoch, n_epoch, idx, len(HR_t), time.time() - step_time, error_ln, error_e))
                    total_e_loss += error_e
                else :
                    error_ln, _, g_and_v = sess.run([ln_loss, update_params_op, grads_and_vars], {LR_for_all_tower : LR_batch, HR_for_all_tower : HR_batch})
                    #write2file(g_and_v, 'grads_and_vars')
                    print("Epoch:[%d/%d]iter:[%d/%d] times: %4.3fs, mse:%.6f" % (epoch, n_epoch, idx, len(HR_t), time.time() - step_time, error_ln))
                
                total_ln_loss += error_ln 
                total_loss += (total_ln_loss + total_e_loss)
                n_iter += 1
                
                if (idx == batch_size):
                  summary_str = sess.run(summary_op, {LR_for_all_tower : LR_batch, HR_for_all_tower : HR_batch})
                  summary_writer.add_summary(summary_str, epoch)
                
            print("Epoch [%d/%d] : times : %4.4fs, loss : %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_loss / n_iter))
            
            if (epoch !=0) and (epoch%ckpt_saving_interval == 0):
                
                npz_file_name = checkpoint_dir+'/generator3d_init_epoch{}.npz'.format(epoch)
                tl.files.save_npz(generator.all_params, name=npz_file_name, sess=sess)
                
                #out = sess.run(generator_test.outputs, {LR_test : test_lr})  
                out = sess.run(generator.outputs, {LR_for_all_tower : test_lr})                          
                write3d(out, test_saving_dir+'test_epoch{}_init.tif'.format(epoch))
                #eval_on_the_fly(epoch, init_training=True)

            
    
    with tf.device('/cpu:0'):
        configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        sess = tf.Session(config=configProto)
        
        sess.run(tf.global_variables_initializer())
        
        if (begin_epoch != 0):
            if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/generator3d_init_epoch{}.npz'.format(begin_epoch), network=generator) is False:
                raise RuntimeError('load and assigened failed : generator3d_init_epoch{}.npz '.format(begin_epoch))
        else:
            #begin_epoch = __find_available_ckpt(n_epoch)
            pass
            
        sess.run(tf.assign(learning_rate_var, learning_rate_init))
        print("learning rate : %f" % learning_rate_init)
        


        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(test_saving_dir)
        tl.files.exists_or_mkdir(valid_saving_dir)

        LR_t = read_all_images(train_lr_img_path, lr_size[0]) 
        HR_t = read_all_images(train_hr_img_path, hr_size[0])
        
        if config.VALID.on_the_fly:
            valid_lr = read_all_images(valid_lr_img_path, lr_size[0])
            write3d(valid_lr, valid_saving_dir+'valid_lr.tif')
        
        test_lr = LR_t[0:batch_size*num_gpus]
        test_hr = HR_t[0:batch_size*num_gpus]
        write3d(test_lr, test_saving_dir+'test_lr.tif')
        write3d(test_hr, test_saving_dir+'test_hr.tif')

        
        __train(begin_epoch, n_epoch+1, LR_t, HR_t, test_lr)
        return
        
def evaluate(epoch, device):
    checkpoint_dir = "checkpoint/"
    lr_size = config.VALID.lr_img_size
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)
    
    start_time = time.time()
    valid_lr_imgs = read_all_images(valid_lr_img_path, lr_size[0])
    
    LR = tf.placeholder("float", [1] + lr_size)
    with tf.device('/gpu:%d' % device):
        generator = res_dense_net(LR, reuse=False)
  
    ckpt_found = False
    filelist = os.listdir(checkpoint_dir)
    for file in filelist:
        if '.npz' in file and str(epoch) in file:
            ckpt = file 
            ckpt_found = True
            break
    if ckpt_found == False:
        raise Exception('no such checkpoint file')
            
        
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt, network=generator)
        
        for idx in range(0,len(valid_lr_imgs)):
            out = sess.run(generator.outputs, {LR : valid_lr_imgs[idx:idx+1]})
            #imageio.volwrite(save_dir+'evaluate_{}.tif'.format(idx), vol=out[0], bigtiff=False)
            write3d(out, save_dir+'evaluate_%06d_epoch%d.tif' % (idx, epoch))
            print('writing %d / %d ...' % (idx + 1, len(valid_lr_imgs)))
    print("time elapsed : %4.4fs " % (time.time() - start_time))
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, eval')
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    ckpt = args.ckpt
    device = args.device

    if args.mode == 'train':
        train(ckpt)
    elif args.mode == 'eval':
        evaluate(ckpt, device)
    else:
        raise Exception("Unknown --mode")
    
