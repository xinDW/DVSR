import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio 

from losses import mean_squared_error, edges_loss, l1_loss
from res_dense_net import res_dense_net
from dbpn import DBPN
from dataset import Dataset
from utils import *
from config import *

batch_size = config.TRAIN.batch_size
lr_size = config.TRAIN.img_size_lr # [depth height width channels]
hr_size = config.TRAIN.img_size_hr

beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch

using_edges_loss = config.TRAIN.using_edges_loss

learning_rate_init = config.TRAIN.learning_rate_init
decay_every = int(n_epoch / 2)
learning_rate_decay = 0.1

checkpoint_dir = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval

test_saving_dir = config.TRAIN.test_saving_path
valid_saving_dir = config.VALID.saving_path                         

train_lr_img_path = config.TRAIN.lr_img_path
train_hr_img_path = config.TRAIN.hr_img_path
train_mr_img_path = config.TRAIN.mr_img_path
valid_lr_img_path = config.VALID.lr_img_path

format_out = True
factor = []
for h, l in zip(hr_size, lr_size):
    factor.append(h // l)
factor = factor[0:3] # remove factor in channels

def get_gradients(grads_and_vars):
    grads = []
    
    for g, _ in grads_and_vars:
        grads.append(g)
    return grads 
    
def tf_print(tensor): 
    input_ = tensor
    data = [tensor]
    return tf.Print(input_=input_, data=data)
    
def train(begin_epoch):

    with tf.variable_scope('learning_rate'):
        learning_rate_var = tf.Variable(learning_rate_init, trainable=False)
        
    LR = tf.placeholder("float", [batch_size] + lr_size)
    MR = tf.placeholder("float", [batch_size] + lr_size)
    if format_out:
        HR = tf.placeholder("float", [batch_size] + hr_size)
    else:
        factors = 1
        for f in factor:
            factors *= f
        HR = tf.placeholder("float", [batch_size] + lr_size[0:3] + [factors])
    
    variable_tag_n1 = 'Resolve'
    variable_tag_n2 = 'Interp'

    with tf.device('/gpu:0'):
        resolver = DBPN(LR, name=variable_tag_n1)
    with tf.device('/gpu:1'):
        #interpolator = interpolator3d(LR, scale=4, is_train=True, reuse=False)
        interpolator = res_dense_net(resolver.outputs, reuse=False, format_out=format_out, name=variable_tag_n2)
        #interpolator = DBPN(LR, name=variable_tag_n2)
        #interpolator = unet3d(LR, reuse=False, is_train=True, name='interpolator')

    resolver.print_params(False)
    interpolator.print_params(False)

    vars_n1 = tl.layers.get_variables_with_name(variable_tag_n1, train_only=True, printable=False)
    vars_n2 = tl.layers.get_variables_with_name(variable_tag_n2, train_only=True, printable=False)

    resolve_loss = mean_squared_error(MR, resolver.outputs, is_mean=True)
    interp_loss = mean_squared_error(HR, interpolator.outputs, is_mean=True)
    #interp_loss = l1_loss(interpolator.outputs, HR)
    
    '''
    #ln_optim = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1).minimize(interp_loss, var_list=vars_n2)
    ln_optim = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1)
    ln_grads_and_vars = ln_optim.compute_gradients(interp_loss, var_list=vars_n2)
    ln_update = ln_optim.apply_gradients(ln_grads_and_vars)
    
    e_loss = 0
    if using_edges_loss:
        e_loss =  (1e-6) * edges_loss(interpolator.outputs, HR) 
        e_optim = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1)#.minimize(e_loss, var_list=vars_n2)
        e_grads_and_var = e_optim.compute_gradients(e_loss, var_list=vars_n2)
        e_update = e_optim.apply_gradients(e_grads_and_var)
        
    t_loss = interp_loss + e_loss
    optim = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1)#.minimize(t_loss, var_list=vars_n2)     
    grads_and_vars = optim.compute_gradients(t_loss, var_list=vars_n2)
    update = optim.apply_gradients(grads_and_vars)
    '''

    t_loss = resolve_loss + interp_loss
    n1_optim = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1).minimize(t_loss, var_list=vars_n1)
    n2_optim = tf.train.AdamOptimizer(learning_rate_var, beta1=beta1).minimize(interp_loss, var_list=vars_n2)

    configProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configProto.gpu_options.allow_growth = True
    sess = tf.Session(config=configProto)
    tl.layers.initialize_global_variables(sess)
    
    tl.files.exists_or_mkdir(checkpoint_dir)
    tl.files.exists_or_mkdir(test_saving_dir)
    tl.files.exists_or_mkdir(valid_saving_dir)

    training_dataset = Dataset(train_hr_img_path, train_lr_img_path, train_mr_img_path, hr_size, lr_size)
    dataset_size = training_dataset.prepare(batch_size, n_epoch - begin_epoch)

    if config.VALID.on_the_fly:
        valid_lr = read_all_images(valid_lr_img_path, lr_size[0])
        write3d(valid_lr, valid_saving_dir+'valid_lr.tif')
    
    test_hr, test_lr, test_mr = training_dataset.for_eval()

    write3d(test_lr, test_saving_dir+'test_lr.tif')
    write3d(test_mr, test_saving_dir+'test_mr.tif')
    if format_out:
        write3d(test_hr, test_saving_dir+'test_hr.tif')
    else:
        write3d(transform(test_hr, factor=factor, inverse=True), test_saving_dir+'test_hr.tif')
    
    
    
    def __find_available_ckpt(end):
        begin = end
        while not os.path.exists(checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin)):
            begin -= 10
            if begin < 0:
                return 0

        print('\n\ninit ckpt found at epoch %d\n\n' % begin)
        load_and_assign_ckpt(sess, checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin), interpolator)                
            
        return begin
        
    def eval_on_the_fly(epoch, init_training=True):
        if config.VALID.on_the_fly:
            out_eval = sess.run(interpolator.outputs, {LR : valid_lr})
            if init_training:
                saving_path = valid_saving_dir+'eval_epoch{}_init.tif'.format(epoch)
            else:
                saving_path = valid_saving_dir+'eval_epoch{}.tif'.format(epoch)
                
            write3d(out_eval, saving_path)

    # tensorflow summary
    tf.summary.scalar('learning_rate', learning_rate_var)        
    tf.summary.scalar('net1_loss', resolve_loss)
    tf.summary.scalar('net2_loss', interp_loss)
    merge_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/', sess.graph)

    if (begin_epoch != 0):
        if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin_epoch), network=interpolator) is False:
            raise RuntimeError('load and assigened failed : {}_epoch{}.npz'.format(label, begin_epoch))
    else:
        begin_epoch = __find_available_ckpt(n_epoch)
        
    sess.run(tf.assign(learning_rate_var, learning_rate_init))
    print("learning rate : %f" % learning_rate_init)
    
    """
    training
    """
    while training_dataset.hasNext():
        step_time = time.time()
        HR_batch, LR_batch, MR_batch, cursor, epoch = training_dataset.iter()

        epoch += begin_epoch
        # adjust learning rate:
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = learning_rate_decay ** (epoch // decay_every)
            sess.run(tf.assign(learning_rate_var, learning_rate_init * new_lr_decay))
            print('\nlearning rate updated : %f\n' % (learning_rate_init * new_lr_decay))

             
        if using_edges_loss:
                #error_ln, error_e, _, = sess.run([interp_loss, e_loss, update], {LR : LR_batch, HR : HR_batch})
                #print("Epoch:[%d/%d] iter:[%d/%d] times: %4.3fs, ln:%.6f, edges:%.6f" % (epoch, n_epoch, cursor, dataset_size, time.time() - step_time, error_ln, error_e))
                pass
        else :
            error_ln, error_t, _,  _, batch_summary = sess.run([interp_loss, t_loss, n1_optim, n2_optim, merge_op], {LR : LR_batch, HR : HR_batch, MR : MR_batch})
            print("Epoch:[%d/%d] iter:[%d/%d] times: %4.3fs, n2:%.6f  total:%.6f" % (epoch, n_epoch, cursor, dataset_size, time.time() - step_time, error_ln, error_t))
            summary_writer.add_summary(batch_summary, epoch * (dataset_size // batch_size - 1) + cursor / batch_size)

        if (epoch !=0) and (epoch%ckpt_saving_interval == 0) and (cursor == batch_size):
            
            n1_npz_file_name = checkpoint_dir + '/{}_resolve_epoch{}.npz'.format(label, epoch)
            n2_npz_file_name = checkpoint_dir+'/{}_interp_epoch{}.npz'.format(label, epoch)
            tl.files.save_npz(resolver.all_params, name=n1_npz_file_name, sess=sess)
            tl.files.save_npz(interpolator.all_params, name=n2_npz_file_name, sess=sess)

            out_resolver, out_interp = sess.run([resolver.outputs, interpolator.outputs], {LR : test_lr})

            if format_out is False:  
                out = transform(out, factor=factor, inverse=True)           
            write3d(out_resolver, test_saving_dir+'test_epoch{}_mr.tif'.format(epoch))
            write3d(out_interp, test_saving_dir+'test_epoch{}_hr.tif'.format(epoch))

        

    '''
    for epoch in range(begin_epoch_init, n_epoch+1):
        total_interp_loss, total_e_loss, n_iter = 0, 0, 0
        epoch_time = time.time()
        for idx in range(batch_size, len(HR_t), batch_size):
            step_time = time.time()
            #HR_batch = tl.prepro.threading_data(HR_t[idx : idx + batch_size], fn=crop_img_fn, size=hr_size)
            #LR_batch = tl.prepro.threading_data(LR_t[idx : idx + batch_size], fn=crop_img_fn, size=lr_size)
            
            if (idx + batch_size > len(HR_t) - 1): 
                idx = idx - (idx + batch_size - len(HR_t) + 1)
                
            HR_batch = HR_t[idx : idx + batch_size]
            LR_batch = LR_t[idx : idx + batch_size]
            
            if using_edges_loss:
                error_ln, error_e, grads, _, = sess.run([interp_loss, e_loss, ln_grads_and_vars, update], {LR : LR_batch, HR : HR_batch})
                #print(get_gradients(grads))
                print("Epoch:[%d/%d]iter:[%d/%d] times: %4.3fs, mse:%.6f, edges:%.6f" % (epoch, n_epoch, idx, len(HR_t), time.time() - step_time, error_ln, error_e))
                total_e_loss += error_e
            else :
                error_ln, grads, _ = sess.run([interp_loss, ln_grads_and_vars, update], {LR : LR_batch, HR : HR_batch})
                #print(get_gradients(grads))
                print("Epoch:[%d/%d]iter:[%d/%d] times: %4.3fs, mse:%.6f" % (epoch, n_epoch, idx, len(HR_t), time.time() - step_time, error_ln))

            total_interp_loss += error_ln 
            
            n_iter += 1
        print("Epoch [%d/%d] : times : %4.4fs, mse : %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_interp_loss / n_iter))
        
        if (epoch !=0) and (epoch%ckpt_saving_interval == 0):
            
            n2_npz_file_name = checkpoint_dir+'/{}_epoch{}.npz'.format(label, epoch)
            tl.files.save_npz(interpolator.all_params, name=n2_npz_file_name, sess=sess)
            
            out = sess.run(interpolator.outputs, {LR : test_lr})   
            if format_out is False:  
                out = transform(out, factor=factor, inverse=True)           
            write3d(out, test_saving_dir+'test_epoch{}_init.tif'.format(epoch))
            
            #eval_on_the_fly(epoch, init_training=True);
    '''
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=0)
    args = parser.parse_args()
    begin_epoch = args.ckpt
    
    
    train(begin_epoch)
    
    
