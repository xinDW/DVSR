import os
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio 

from losses import mean_squared_error, edges_loss, l1_loss
from model import res_dense_net, DBPN
from dataset import Dataset
from utils import *
from config import config

batch_size = config.TRAIN.batch_size
lr_size = config.TRAIN.img_size_lr # [depth height width channels]
hr_size = config.TRAIN.img_size_hr

label = config.label
device_id = config.TRAIN.device_id
conv_kernel = config.TRAIN.conv_kernel
using_batch_norm = config.TRAIN.using_batch_norm 

beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch

learning_rate_init = config.TRAIN.learning_rate_init
decay_every = int(n_epoch / 2)
learning_rate_decay = 0.1

checkpoint_dir = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval
test_saving_dir = config.TRAIN.test_saving_path
log_dir = config.TRAIN.log_dir

factor = []
for h, l in zip(hr_size, lr_size):
    factor.append(h // l)
factor = factor[0:3] # remove factor in channels dimension

def get_gradients(grads_and_vars):
    grads = []
    
    for g, _ in grads_and_vars:
        grads.append(g)
    return grads 
    
def tf_print(tensor): 
    input_ = tensor
    data = [tensor]
    return tf.Print(input_=input_, data=data)

class Trainer:
    """
    Params:
        -architechture: ['2stage_interp_first', '2stage_resolve_first', 'rdn']
    """
    def __init__(self, dataset, architecture='2stage_resolve_first'):
        assert architecture in ['2stage_interp_first', '2stage_resolve_first', 'rdn'] 
        self.archi = architecture
        self.dataset = dataset
        self.loss = {}
        self.test_loss = {}
        self.optim = {}

    def build_graph(self):
        with tf.variable_scope('learning_rate'):
            self.learning_rate_var = tf.Variable(learning_rate_init, trainable=False)
        
        self.LR = tf.placeholder("float", [batch_size] + lr_size, name="LR")       
        self.HR = tf.placeholder("float", [batch_size] + hr_size, name="HR")

        if ('2stage' in self.archi):
            variable_tag_n1 = 'Resolve'
            variable_tag_n2 = 'Interp'

            if ('resolve_first' in self.archi):
                self.MR = tf.placeholder("float", [batch_size] + lr_size, name="MR")  
                with tf.device('/gpu:%d' % device_id):
                    resolver = DBPN(self.LR, upscale=False, name=variable_tag_n1)
                #with tf.device('/gpu:1'):
                    interpolator = res_dense_net(resolver.outputs, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=True, name=variable_tag_n2)

                training_loss_resolve = mean_squared_error(self.MR, resolver.outputs, is_mean=True)
                training_loss_interp = mean_squared_error(self.HR, interpolator.outputs, is_mean=True)

                resolver_test = DBPN(self.LR, upscale=False, reuse=True, name=variable_tag_n1)
                interpolator_test = res_dense_net(resolver_test.outputs, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, reuse=True, name=variable_tag_n2)
               
                test_loss_resolve = mean_squared_error(self.MR, resolver_test.outputs, is_mean=True)
                test_loss_interp = mean_squared_error(self.HR, interpolator_test.outputs, is_mean=True)
                test_loss = test_loss_interp + test_loss_resolve

            else :
                self.MR = tf.placeholder("float", [batch_size] + hr_size)   
                with tf.device('/gpu:1'):
                    interpolator = DBPN(self.LR, upscale=True, name=variable_tag_n1)
                with tf.device('/gpu:0'):
                    resolver = res_dense_net(interpolator.outputs, factor=1, conv_kernel=conv_kernel, reuse=False, bn=using_batch_norm, is_train=True, name=variable_tag_n2)

                training_loss_resolve = mean_squared_error(self.HR, resolver.outputs, is_mean=True)
                training_loss_interp = mean_squared_error(self.MR, interpolator.outputs, is_mean=True)    

            resolver.print_params(details=False)
            interpolator.print_params(details=False)

            vars_n1 = tl.layers.get_variables_with_name(variable_tag_n1, train_only=True, printable=False)
            vars_n2 = tl.layers.get_variables_with_name(variable_tag_n2, train_only=True, printable=False)
            
            #training_loss_resolve = mean_squared_error(self.MR, resolver.outputs, is_mean=True)
            #training_loss_interp = mean_squared_error(self.HR, interpolator.outputs, is_mean=True)
            training_loss = training_loss_resolve + training_loss_interp

            #n1_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(training_loss, var_list=vars_n1)
            #n2_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(training_loss_interp, var_list=vars_n2)
            n1_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(training_loss_interp)
            n2_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(training_loss_resolve)
            n_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(training_loss)

            self.loss.update({'training_loss' : training_loss, 'training_loss_interp' : training_loss_interp, 'training_loss_resolve' : training_loss_resolve})
            self.test_loss.update({'test_loss' : test_loss, 'test_loss_interp' : test_loss_interp, 'test_loss_resolve' : test_loss_resolve})
            self.optim.update({'n1_optim' : n1_optim, 'n2_optim' : n2_optim, 'n_optim' : n_optim})

            self.resolver = resolver
            self.interpolator = interpolator

        else : 
            variable_tag = 'rdn'
            
            with tf.device('/gpu:1'):
                net = res_dense_net(self.LR, reuse=False, name=variable_tag)

            net_vars = tl.layers.get_variables_with_name(variable_tag, train_only=True, printable=False)

            l2_loss = mean_squared_error(self.HR, net.outputs, is_mean=True)

            l2_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(l2_loss, var_list=net_vars)

            
            self.loss.update({'l2_loss' : l2_loss})
            self.optim.update({'l2_optim' : l2_optim})

            
    def _find_available_ckpt(self, end, sess):
        begin = end
        while not os.path.exists(checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin)):
            begin -= 10
            if begin < 0:
                return 0

        print('\n\ninit ckpt found at epoch %d\n\n' % begin)
        load_and_assign_ckpt(sess, checkpoint_dir+'/{}_resolve_epoch{}.npz'.format(label, begin), self.resolver)                
        load_and_assign_ckpt(sess, checkpoint_dir+'/{}_interp_epoch{}.npz'.format(label, begin), self.interpolator)  
        return begin

    def _valid_on_the_fly(self, sess, epoch, batch_idx, valid_lr_batch, init_training=False):
        out_valid = sess.run(self.interpolator.outputs, {self.LR : valid_lr_batch})
        if init_training:
            saving_path = test_saving_dir+'valid_epoch{}-{}_init.tif'.format(epoch, batch_idx)
        else:
            saving_path = test_saving_dir+'valid_epoch{}-{}.tif'.format(epoch, batch_idx)
            
        write3d(out_valid, saving_path)

    def _load_designated_ckpt(self, begin_epoch, sess):
        if (begin_epoch != 0):
            interp_ckpt_found = False
            resolve_ckpt_found = False
            filelist = os.listdir(checkpoint_dir)
            for file in filelist:
                if '.npz' in file and str(begin_epoch) in file:
                    if 'interp' in file:
                        interp_ckpt = file 
                        interp_ckpt_found = True
                    if 'resolve' in file:
                        resolve_ckpt = file
                        resolve_ckpt_found = True
                    if interp_ckpt_found and resolve_ckpt_found:
                        break

            if not interp_ckpt_found: 
                raise Exception('designated checkpoint file for interpolator not found')
            if not resolve_ckpt_found:
                raise Exception('designated checkpoint file for resolver not found')
            load_resolver = load_and_assign_ckpt(sess, checkpoint_dir+'/{}'.format(resolve_ckpt), self.resolver)                
            load_interp = load_and_assign_ckpt(sess, checkpoint_dir+'/{}'.format(interp_ckpt), self.interpolator)  
            if not load_resolver or not load_interp:
                raise RuntimeError('load and assigened ckpt failed')
            return begin_epoch
        else:
            return self._find_available_ckpt(n_epoch, sess)

    def _save_intermediate_ckpt(self, epoch, sess):
        n1_npz_file_name = checkpoint_dir + '/{}_resolve_epoch{}.npz'.format(label, epoch)
        n2_npz_file_name = checkpoint_dir+'/{}_interp_epoch{}.npz'.format(label, epoch)
        tl.files.save_npz(self.resolver.all_params, name=n1_npz_file_name, sess=sess)
        tl.files.save_npz(self.interpolator.all_params, name=n2_npz_file_name, sess=sess)

        if config.VALID.on_the_fly:
            for idx in range(0, len(self.valid_lr), batch_size):
                if idx + batch_size <= len(self.valid_lr):
                    valid_lr_batch = self.valid_lr[idx : idx + batch_size]
                    self._valid_on_the_fly(sess, epoch, idx, valid_lr_batch)

    def _make_summaries(self, sess):
        # tensorflow summary
        tf.summary.scalar('learning_rate', self.learning_rate_var) 
        for name, loss in self.loss.items():       
            tf.summary.scalar(name, loss)
        for name, loss in self.test_loss.items():       
            tf.summary.scalar(name, loss)

        self.summary_op = tf.summary.merge_all()

        self.training_loss_writer = tf.summary.FileWriter(log_dir + 'training', sess.graph)
        self.test_loss_writer = tf.summary.FileWriter(log_dir + 'test')

    def train(self, begin_epoch=0):
        
        configProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        sess = tf.Session(config=configProto)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(self.learning_rate_var, learning_rate_init))

        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(test_saving_dir)

        training_dataset = self.dataset
        n_training_pairs = training_dataset.prepare(batch_size, n_epoch - begin_epoch)

        if config.VALID.on_the_fly:
            self.valid_lr = self.dataset.for_valid()
            write3d(self.valid_lr, test_saving_dir+'valid_lr.tif')
        
        self.test_hr, self.test_lr, self.test_mr = training_dataset.for_test()

        write3d(self.test_lr, test_saving_dir+'test_lr.tif')
        write3d(self.test_hr, test_saving_dir+'test_hr.tif')
        if ('2stage' in self.archi):
            write3d(self.test_mr, test_saving_dir+'test_mr.tif')
        
        self._make_summaries(sess)
        
        begin_epoch = self._load_designated_ckpt(begin_epoch, sess)   
            
        
        """
        training
        """
        fetches = dict(self.loss, **(self.optim))
        fetches['batch_summary'] = self.summary_op

        while training_dataset.hasNext():
            step_time = time.time()
            HR_batch, LR_batch, MR_batch, cursor, epoch = training_dataset.iter()

            epoch += begin_epoch
            # adjust learning rate:
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = learning_rate_decay ** (epoch // decay_every)
                sess.run(tf.assign(self.learning_rate_var, learning_rate_init * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (learning_rate_init * new_lr_decay))

            evaluated = sess.run(fetches, {self.LR : LR_batch, self.HR : HR_batch, self.MR : MR_batch})
            print("Epoch:[%d/%d] iter:[%d/%d] times: %4.3fs" % (epoch, n_epoch, cursor + 1, n_training_pairs, time.time() - step_time))
            losses_val = {name : value for name, value in evaluated.items() if 'loss' in name}
            print(losses_val)

            n_iters_passed = epoch * (n_training_pairs // batch_size) + cursor / batch_size
            self.training_loss_writer.add_summary(evaluated['batch_summary'], n_iters_passed)

            if (epoch !=0) and (epoch%ckpt_saving_interval == 0) and (cursor == n_training_pairs - 1):
                self._save_intermediate_ckpt(epoch, sess)
                for idx in range(0, len(self.test_lr), batch_size):
                    if idx + batch_size < len(self.test_lr):
                        test_lr_batch = self.test_lr[idx : idx + batch_size]
                        test_hr_batch = self.test_hr[idx : idx + batch_size]
                        test_mr_batch = self.test_mr[idx : idx + batch_size]

                        out_resolver, out_interp, summary_t_loss = sess.run([self.resolver.outputs, self.interpolator.outputs, self.summary_op], 
                            {self.LR : test_lr_batch, self.HR : test_hr_batch, self.MR : test_mr_batch})
                        write3d(out_resolver, test_saving_dir+'mr_test_epoch{}_{}.tif'.format(epoch, idx))
                        write3d(out_interp, test_saving_dir+'hr_test_epoch{}_{}.tif'.format(epoch, idx))
                self.test_loss_writer.add_summary(summary_t_loss, n_iters_passed)
          
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=0)
    args = parser.parse_args()
    begin_epoch = args.ckpt

    train_lr_path = config.TRAIN.lr_img_path
    train_hr_path = config.TRAIN.hr_img_path
    train_mr_path = config.TRAIN.mr_img_path
    valid_lr_path = config.VALID.lr_img_path if config.VALID.on_the_fly else None # lr measuremnet for validation during the training   
    test_lr_path = train_lr_path + 'test/'
    test_hr_path = train_hr_path + 'test/'
    test_mr_path = train_mr_path + 'test/'

    mr_size = lr_size if config.archi == '2stage_resolve_first' else hr_size

    dataset = Dataset(lr_size, hr_size, lr_size, 
        train_lr_path, train_hr_path, train_mr_path, 
        test_lr_path, test_hr_path, test_mr_path, valid_lr_path)

    trainer = Trainer(dataset, architecture=config.archi)
    trainer.build_graph()
    trainer.train(begin_epoch)
    
                  

    
    
