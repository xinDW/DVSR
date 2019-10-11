import os
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio 

from model import res_dense_net, DBPN, unet3d
from model.losses import l2_loss, edges_loss, img_gradient_loss, l1_loss
from dataset import Dataset
from utils import *
from config import config
from train import Trainer

batch_size = config.TRAIN.batch_size
lr_size = config.TRAIN.img_size_lr # [depth height width channels]
hr_size = config.TRAIN.img_size_hr

label = config.label
device_id = config.TRAIN.device_id
gpu_num = config.TRAIN.num_gpus
conv_kernel = config.TRAIN.conv_kernel
using_batch_norm = config.TRAIN.using_batch_norm 
using_edge_loss = config.TRAIN.using_edge_loss
using_grad_loss = config.TRAIN.using_grad_loss

beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch

learning_rate_init = config.TRAIN.learning_rate_init
decay_every = config.TRAIN.decay_every 
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

class MutilDeviceTrainer(Trainer):
    """
    Params:
        -architechture: ['2stage_interp_first', '2stage_resolve_first', 'rdn', 'unet', 'dbpn']
    """
    def __init__(self, dataset, architecture='2stage_resolve_first', visualize_features=False):
        Trainer.__init__(self, dataset, architecture, visualize_features)
    

    def build_graph(self):
        assert batch_size % gpu_num == 0
        tower_batch = batch_size // gpu_num
        
        with tf.device('/cpu:0'):
            self.learning_rate_var = tf.Variable(learning_rate_init, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1)
            tower_grads = []

            self.plchdr_lr = tf.placeholder("float", [batch_size] + lr_size, name="LR")       
            self.plchdr_hr = tf.placeholder("float", [batch_size] + hr_size, name="HR")
            if ('2stage' in self.archi):
                if ('resolve_first' in self.archi):
                    self.plchdr_mr = tf.placeholder("float", [batch_size] + lr_size, name="MR")  
                else:
                    self.plchdr_mr = tf.placeholder("float", [batch_size] + hr_size, name='MR')  

            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(gpu_num):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % i) as name_scope:
                            if ('2stage' in self.archi):
                                variable_tag_res = 'Resolve'
                                variable_tag_interp = 'Interp'
                                if ('resolve_first' in self.archi):
                                    var_tag_n2 = variable_tag_interp
                                    net_stage1 = DBPN(self.plchdr_lr[i * tower_batch : (i + 1) * tower_batch], upscale=False, name=variable_tag_res)
                                    net_stage2 = res_dense_net(net_stage1.outputs, factor=config.factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=True, name=variable_tag_interp)
                                    self.resolver = net_stage1
                                    self.interpolator = net_stage2
                                else :
                                    var_tag_n2 = variable_tag_res
                                    net_stage1 = res_dense_net(self.plchdr_lr[i * tower_batch : (i + 1) * tower_batch], factor=config.factor, conv_kernel=conv_kernel, reuse=False, bn=using_batch_norm, is_train=True, name=variable_tag_interp)
                                    net_stage2 = DBPN(net_stage1.outputs, upscale=False, name=variable_tag_res)
                                    self.resolver = net_stage2
                                    self.interpolator = net_stage1
                                net_stage1.print_params(details=False)
                                net_stage2.print_params(details=False)

                                #vars_n1 = tl.layers.get_variables_with_name(variable_tag_res, train_only=True, printable=False)
                                vars_n2 = tl.layers.get_variables_with_name(var_tag_n2, train_only=True, printable=False)
                                
                                loss_training_n1 = l2_loss(self.plchdr_mr[i * tower_batch : (i + 1) * tower_batch], net_stage1.outputs)
                                loss_training_n2 = l2_loss(self.plchdr_hr[i * tower_batch : (i + 1) * tower_batch], net_stage2.outputs)
                                
                                loss_training = loss_training_n1 + loss_training_n2
                                tf.add_to_collection('losses', loss_training)
                                loss_tower = tf.add_n(tf.get_collection('losses', name_scope)) # the total loss for the current tower

                                grads = optimizer.compute_gradients(loss_tower)
                                tower_grads.append(grads)

                                self.loss.update({'loss_training' : loss_training, 'loss_training_n2' : loss_training_n2, 'loss_training_n1' : loss_training_n1})
                                

                                if using_edge_loss:
                                    loss_edges = edges_loss(net_stage2.outputs, self.plchdr_hr[i * tower_batch : (i + 1) * tower_batch])
                                    e_optim = optimizer.minimize(loss_edges, var_list=vars_n2)
                                    self.loss.update({'edge_loss' : loss_edges})
                                    self.optim.update({'e_optim' : e_optim})

                                if using_grad_loss:
                                    loss_grad = img_gradient_loss(net_stage2.outputs, self.plchdr_hr[i * tower_batch : (i + 1) * tower_batch])
                                    g_optim = optimizer.minimize(loss_grad, var_list=vars_n2)
                                    self.loss.update({'grad_loss' : loss_grad})
                                    self.optim.update({'g_optim' : g_optim})

                            else : 
                                variable_tag = '1stage_%s' % self.archi
                                if self.archi is 'rdn':
                                    net = res_dense_net(self.plchdr_lr[i * tower_batch : (i + 1) * tower_batch], factor=config.factor, reuse=i > 0, name=variable_tag)
                                elif self.archi is 'unet':
                                    net = unet3d(self.plchdr_lr[i * tower_batch : (i + 1) * tower_batch], upscale=True, reuse=i > 0, is_train=True, name=variable_tag)
                                elif self.archi is 'dbpn':
                                    net = DBPN(self.plchdr_lr[i * tower_batch : (i + 1) * tower_batch], upscale=True, reuse=i > 0, name=variable_tag)
                                else:
                                     raise Exception('unknow architecture: %s' % self.archi)

                                
                                if i == 0:
                                    self.net = net
                                    
                                ln_loss = l2_loss(self.plchdr_hr[i * tower_batch : (i + 1) * tower_batch], net.outputs)
                                tf.add_to_collection('losses', ln_loss)
                                loss_tower = tf.add_n(tf.get_collection('losses', name_scope)) # the total loss for the current tower

                                grads = optimizer.compute_gradients(loss_tower)
                                tower_grads.append(grads)
                                
                                self.loss.update({'ln_loss' : ln_loss})

                                '''
                                if using_edge_loss:
                                    loss_edges = edges_loss(net.outputs, self.plchdr_hr[i * tower_batch : (i + 1) * tower_batch])
                                    e_optim = optimizer.minimize(loss_edges, var_list=net_vars)
                                    self.loss.update({'edge_loss' : loss_edges})
                                    self.optim.update({'e_optim' : e_optim})
                                if using_grad_loss:
                                    loss_grad = img_gradient_loss(net.outputs, self.plchdr_hr[i * tower_batch : (i + 1) * tower_batch])
                                    g_optim = optimizer.minimize(loss_grad, var_list=net_vars)
                                    self.loss.update({'grad_loss' : loss_grad})
                                    self.optim.update({'g_optim' : g_optim})
                                '''

                            tf.get_variable_scope().reuse_variables()

            grads = self._average_gradient(tower_grads)
            n_optim = optimizer.apply_gradients(grads)
            self.optim.update({'n_optim' : n_optim})    
        
    def _average_gradient(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
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

    '''
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
        out_valid = sess.run(self.interpolator.outputs, {self.plchdr_lr : valid_lr_batch})
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
        for name, loss in self.loss_test.items():       
            tf.summary.scalar(name, loss)

        self.summary_op = tf.summary.merge_all()

        self.training_loss_writer = tf.summary.FileWriter(log_dir + 'training', sess.graph)
        self.test_loss_writer = tf.summary.FileWriter(log_dir + 'test')

    def _adjust_learning_rate(self, epoch, sess):
        new_lr_decay = learning_rate_decay ** (epoch // decay_every)
        sess.run(tf.assign(self.learning_rate_var, learning_rate_init * new_lr_decay))
        print('\nlearning rate updated : %f\n' % (learning_rate_init * new_lr_decay))
    '''

    '''
    def train(self, begin_epoch=0, print_loss=True):
        
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
        
        #begin_epoch = self._load_designated_ckpt(begin_epoch, sess)   
            
        
        """
        training
        """
        fetches = dict(self.loss, **(self.optim))
        fetches['batch_summary'] = self.summary_op

        while training_dataset.hasNext():
            step_time = time.time()
            HR_batch, LR_batch, MR_batch, cursor, epoch = training_dataset.iter()

            epoch += begin_epoch

            if ('2stage' in self.archi):
                evaluated = sess.run(fetches, {self.plchdr_lr : LR_batch, self.plchdr_hr : HR_batch, self.plchdr_mr : MR_batch})
            else:
                evaluated = sess.run(fetches, {self.plchdr_lr : LR_batch, self.plchdr_hr : HR_batch})
            print("Epoch:[%d/%d] iter:[%d/%d] times: %4.3fs" % (epoch, n_epoch, cursor + 1, n_training_pairs, time.time() - step_time))
            
            if print_loss:
                #losses_val = {name : value for name, value in evaluated.items() if 'loss' in name}
                #print(losses_val)
                for name, value in evaluated.items():
                    if 'loss' in name:
                        print('%s : %.7f\t' % (name, value))

            n_iters_passed = epoch * (n_training_pairs // batch_size) + cursor / batch_size
            self.training_loss_writer.add_summary(evaluated['batch_summary'], n_iters_passed)

            if (epoch != 0 and epoch % decay_every == 0 and cursor == n_training_pairs - 1 ):
                self._adjust_learning_rate(epoch, sess)

            if (epoch !=0) and (epoch%ckpt_saving_interval == 0) and (cursor == n_training_pairs - 1):
                self._save_intermediate_ckpt(epoch, sess)
                for idx in range(0, len(self.test_lr), batch_size):
                    if idx + batch_size < len(self.test_lr):
                        test_lr_batch = self.test_lr[idx : idx + batch_size]
                        test_hr_batch = self.test_hr[idx : idx + batch_size]
                        test_mr_batch = self.test_mr[idx : idx + batch_size]

                        out_resolver, out_interp, summary_t_loss = sess.run([self.resolver.outputs, self.interpolator.outputs, self.summary_op], 
                            {self.plchdr_lr : test_lr_batch, self.plchdr_hr : test_hr_batch, self.plchdr_mr : test_mr_batch})
                        write3d(out_resolver, test_saving_dir+'mr_test_epoch{}_{}.tif'.format(epoch, idx))
                        write3d(out_interp, test_saving_dir+'sr_test_epoch{}_{}.tif'.format(epoch, idx))
                self.test_loss_writer.add_summary(summary_t_loss, n_iters_passed)
    '''  
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    args = parser.parse_args()
    begin_epoch = args.ckpt

    train_lr_path = config.TRAIN.lr_img_path
    train_hr_path = config.TRAIN.hr_img_path
    train_mr_path = config.TRAIN.mr_img_path if '2stage' in config.archi else None
    valid_lr_path = config.TRAIN.valid_lr_path # lr measuremnet for validation during the training   

    test_data_dir = config.TRAIN.test_data_path
    if test_data_dir is not None:
        test_lr_path = test_data_dir + 'lr/'
        test_hr_path = test_data_dir + 'hr/'
        test_mr_path = test_data_dir +  'mr/'
    else:
        test_lr_path = None
        test_hr_path = None
        test_mr_path = None

    mr_size = lr_size #if config.archi == '2stage_resolve_first' else hr_size

    dataset = Dataset(lr_size, hr_size,  
        train_lr_path, train_hr_path, test_lr_path, test_hr_path,
        mr_size, train_mr_path, test_mr_path, valid_lr_path)

    trainer = MutilDeviceTrainer(dataset, architecture=config.archi)
    trainer.build_graph()
    trainer.train(begin_epoch, test=False)
    
                  

    
    
