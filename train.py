import os
import time
import re
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio 
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import res_dense_net, DBPN, resnet, unet3d, denoise_net, fusedSegNet, load_ckpt_partial
from model.losses import l2_loss, edges_loss, img_gradient_loss, l1_loss
from dataset import Dataset
from utils import *
from config import config

batch_size = config.TRAIN.batch_size
lr_size = config.TRAIN.img_size_lr # [depth height width channels]
hr_size = config.TRAIN.img_size_hr

label = config.label
device_id = config.TRAIN.device_id
conv_kernel = config.TRAIN.conv_kernel
loss_fn = l2_loss if config.loss == 'mse' else l1_loss

using_batch_norm = config.using_batch_norm 
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
valid_saving_dir = test_saving_dir + 'valid_otf/'
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

class BaseTrainer:
    """
    Params:
        -archi1: [None, 'dbpn', 'denoise']
            architecture of the stage1 sub-net. None if 1-stage is expected.
        -archi2: ['rdn', 'unet', 'dbpn']
            architecture of the stage2 sub-net. 
        _visualize_features: if True, save all the feature maps of the test set into tiff
    """
    def __init__(self, dataset, archi1='dbpn', archi2='rdn', visualize_features=False, pretrain=False):
        archi1 in [None, 'dbpn', 'denoise'] or _raise(ValueError('illegal argument archi1: %s' % archi1))
        archi2 in ['rdn'] or _raise(ValueError('illegal argument archi2: %s' % archi2))

        self.archi1           = archi1
        self.archi2           = archi2
        self.dataset          = dataset
        self.valid_on_the_fly = dataset.hasValidation
        self.visualize        = visualize_features

        self.nets      = {}
        self.loss      = {}
        self.loss_test = {}
        self.optim     = {}
        self.pretrain  = pretrain

        self.loss_test_plt = [] # plotting test loss using matplotlib
        

    def build_graph(self):
        with tf.variable_scope('learning_rate'):
            self.learning_rate_var = tf.Variable(learning_rate_init, trainable=False)
        
        self.plchdr_lr = tf.placeholder("float", [batch_size] + lr_size, name="LR")       
        self.plchdr_hr = tf.placeholder("float", [batch_size] + hr_size, name="HR")

            
    def _find_available_ckpt(self, end, sess):
        pass

    def _valid_on_the_fly(self, sess, epoch, batch_idx, valid_lr_batch, init_training=False):
        out_valid = sess.run(self.interpolator.outputs, {self.plchdr_lr : valid_lr_batch})
        if init_training:
            saving_path = valid_saving_dir+'valid_epoch{}-{}_init.tif'.format(epoch, batch_idx)
        else:
            saving_path = valid_saving_dir+'valid_epoch{}-{}.tif'.format(epoch, batch_idx)
            
        write3d(out_valid, saving_path, scale_pixel_value=True)

    def _traversal_through_ckpts(self, checkpoint_dir, epoch, label=None):
            ckpt_found = False
            filelist = os.listdir(checkpoint_dir)
            for file in filelist:
                if '.npz' in file and str(epoch) in file:
                    if label is not None:
                        if label in file:
                            return file
                    else:
                        return file
            return None

    def _load_designated_ckpt(self, begin_epoch, sess):
        

        if (begin_epoch != 0):
            if (self.archi1 is not None):
                resolve_ckpt = self._traversal_through_ckpts(checkpoint_dir, begin_epoch, 'resolve')
                interp_ckpt  = self._traversal_through_ckpts(checkpoint_dir, begin_epoch, 'interp')
                
                interp_ckpt is not None or _raise(Exception('designated checkpoint file for interpolator not found'))
                resolve_ckpt is not None or _raise(Exception('designated checkpoint file for resolver not found'))

                load_resolver = load_and_assign_ckpt(sess, checkpoint_dir+'/{}'.format(resolve_ckpt), self.resolver)                
                load_interp   = load_and_assign_ckpt(sess, checkpoint_dir+'/{}'.format(interp_ckpt), self.interpolator)  
                
                (load_resolver and load_interp) or _raise(RuntimeError('load and assigened ckpt failed'))

                return begin_epoch

            else:
                ckpt = self._traversal_through_ckpts(checkpoint_dir, begin_epoch)
                
                ckpt is not None or _raise(Exception('[!] designated checkpoint file not found'))
                load_and_assign_ckpt(sess, checkpoint_dir+'/{}'.format(ckpt), self.net) or _raise(RuntimeError('load and assigened ckpt failed'))
                
                return begin_epoch

        else:
            #return self._find_available_ckpt(n_epoch, sess)
            return 0

    def _save_intermediate_ckpt(self, tag, sess):
        print('')
        tag = ('epoch%d' % tag) if is_number(tag) else tag

        if (self.archi1 is not None):
            n1_npz_file_name = checkpoint_dir + '/resolve_{}.npz'.format(tag)
            n2_npz_file_name = checkpoint_dir + '/interp_{}.npz'.format(tag)
            tl.files.save_npz(self.resolver.all_params, name=n1_npz_file_name, sess=sess)
            tl.files.save_npz(self.interpolator.all_params, name=n2_npz_file_name, sess=sess)
        else:
            npz_file_name = checkpoint_dir + '/{}.npz'.format(tag)
            tl.files.save_npz(self.net.all_params, name=npz_file_name, sess=sess)

        # if self.valid_on_the_fly:
        #     for idx in range(0, len(self.valid_lr), batch_size):
        #         if idx + batch_size <= len(self.valid_lr):
        #             valid_lr_batch = self.valid_lr[idx : idx + batch_size]
        #             self._valid_on_the_fly(sess, epoch, idx, valid_lr_batch)

    def _get_output_node_name(self, sess):
        name = None
        for op in sess.graph.get_operations():
            if 'out' in op.name:
                name = op.name
        name or __raise(ValueError('ops with name "out" is not in the default graph'))
        print('\noutput node found : %s\n' % name)
        return name 

    def _save_pb(self, sess):
        if 'output_node' not in dir(self):
            self.output_node = self._get_output_node_name(sess)

        pb_file = checkpoint_dir + '/best_model.pb'
        save_graph_as_pb(sess=sess, output_node_names=self.output_node, output_graph_file=pb_file)

    def _adjust_learning_rate(self, epoch, sess):
        new_lr_decay = learning_rate_decay ** (epoch // decay_every)
        sess.run(tf.assign(self.learning_rate_var, learning_rate_init * new_lr_decay))
        print('\nlearning rate updated : %f\n' % (learning_rate_init * new_lr_decay))

    def _visualize_layers(self, sess, final_layer, feed_dict):
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
            save_path = test_saving_dir + 'layers/%03d/' % layer_idx
            tl.files.exists_or_mkdir(save_path)
            filename = save_path +'{}.tif'.format(name)
            write3d(feat, filename)
            layer_idx += 1


    def _make_summaries(self, sess):
        # tensorflow summary
        tf.summary.scalar('learning_rate', self.learning_rate_var) 
        training_summary_protbufs = []
        test_summary_protbufs = []
        for name, loss in self.loss.items():  
            print('add loss : [%s] to summaries' % name)     
            training_summary_protbufs.append(tf.summary.scalar(name, loss))
        for name, loss in self.loss_test.items(): 
            print('add test loss : [%s] to summaries' % name)           
            test_summary_protbufs.append(tf.summary.scalar(name, loss))


        self.summary_op = tf.summary.merge_all()
        self.summary_op_training = tf.summary.merge(training_summary_protbufs)
        
        self.summary_op_test     = tf.summary.merge(test_summary_protbufs) if len(test_summary_protbufs) > 0 else None

        self.training_loss_writer = tf.summary.FileWriter(log_dir + 'training', sess.graph)
        self.test_loss_writer     = tf.summary.FileWriter(log_dir + 'test')


    def _record_avg_test_loss(self, epoch, sess):
        if 'min_test_loss' not in dir(self):
            self.min_test_loss = 1e10

        test_loss = 0
        test_data_num = len(self.test_lr)
        for idx in range(0, test_data_num, batch_size):
            if idx + batch_size <= test_data_num:
                test_lr_batch = self.test_lr[idx : idx + batch_size]
                test_hr_batch = self.test_hr[idx : idx + batch_size]
                if (self.archi1 is not None): 
                    test_mr_batch = self.test_mr[idx : idx + batch_size]
                    feed_test = {self.plchdr_lr : test_lr_batch, self.plchdr_hr : test_hr_batch, self.plchdr_mr : test_mr_batch}
                    name = 'loss_test_n2'
                else:
                    feed_test = {self.plchdr_lr : test_lr_batch, self.plchdr_hr : test_hr_batch}
                    name = 'ln_loss_test'
                
                test_loss_batch = sess.run(self.loss_test, feed_test)
                test_loss += test_loss_batch[name]
                print('\rvalidation [% 2d/% 2d] loss = %.6f   ' % (idx, test_data_num, test_loss_batch[name]), end='')


        test_loss /= (len(self.test_lr) // batch_size)       
        
        if (test_loss < self.min_test_loss):
            self.min_test_loss = test_loss
            self._save_intermediate_ckpt(tag='best', sess=sess)
            # self._save_pb(sess)

        print('avg = %.6f best = %.6f' % (test_loss, self.min_test_loss))
        self.loss_test_plt.append([epoch, test_loss])



    def _add_test_summary(self, iter, epoch, sess):
        local_iter = 0

        # for name, loss in self.loss_test.items(): 
        #     print('add test loss : [%s] to summaries' % name)           
        #     test_summary_protbufs.append(tf.summary.scalar(name, loss))
        
        for idx in range(0, len(self.test_lr), batch_size):
            if idx + batch_size <= len(self.test_lr):
                test_lr_batch = self.test_lr[idx : idx + batch_size]
                test_hr_batch = self.test_hr[idx : idx + batch_size]
                if (self.archi1 is not None): 
                    test_mr_batch = self.test_mr[idx : idx + batch_size]

                    feed_test = {self.plchdr_lr : test_lr_batch, self.plchdr_hr : test_hr_batch, self.plchdr_mr : test_mr_batch}
                    out_resolver, out_interp, summary_t_loss = sess.run([self.resolver.outputs, self.interpolator.outputs, self.summary_op_test], feed_test)
                    
                    if idx == 0:
                        write3d(out_resolver, test_saving_dir+'mr_test_epoch{}_{}.tif'.format(epoch, idx))
                        write3d(out_interp, test_saving_dir+'sr_test_epoch{}_{}.tif'.format(epoch, idx))

                    self.test_loss_writer.add_summary(summary_t_loss, iter + local_iter)
                    
                    if self.visualize and idx == 0:
                        self._visualize_layers(sess, self.interpolator, feed_test)
                        self._visualize_layers(sess, self.resolver, feed_test)
                else:
                    feed_test = {self.plchdr_lr : test_lr_batch, self.plchdr_hr : test_hr_batch}
                    out, summary_t_loss = sess.run([self.net.outputs, self.summary_op_test], feed_test)

                    if idx == 0:
                        write3d(out, test_saving_dir+'sr_test_epoch{}_{}.tif'.format(epoch, idx))

                    if self.visualize and idx == 0:
                        self._visualize_layers(sess, self.net, feed_test)

                    self.test_loss_writer.add_summary(summary_t_loss, iter + local_iter)

                local_iter += 1

    def _get_test_data(self):
        tl.files.exists_or_mkdir(test_saving_dir)
        self.test_hr, self.test_lr, self.test_mr = self.dataset.for_test()

        write3d(self.test_lr[0:batch_size], test_saving_dir+'test_lr.tif')
        write3d(self.test_hr[0:batch_size], test_saving_dir+'test_hr.tif')

    def _get_valid_otf_data(self):
        if self.valid_on_the_fly:
            self.valid_lr = self.dataset.for_valid()
            tl.files.exists_or_mkdir(valid_saving_dir)
            write3d(self.valid_lr, valid_saving_dir+'valid_lr.tif')   
    
    def _plot_test_loss(self):
        loss = np.asarray(self.loss_test_plt)
        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        plt.show()
        plt.savefig(test_saving_dir + 'test_loss.png', bbox_inches='tight')

    def _save_history(self):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = 'test'

        loss = self.loss_test_plt
        [ws.append(loss[i]) for i in range(len(loss))]
        wb.save(test_saving_dir + "%s.xlsx" % label)


    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._save_history()
            self._plot_test_loss()

    def _train(self, begin_epoch=0, test=True, verbose=True):
        
        configProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        sess = tf.Session(config=configProto)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(self.learning_rate_var, learning_rate_init))

        tl.files.exists_or_mkdir(checkpoint_dir)
        

        training_dataset = self.dataset
        n_training_pairs = training_dataset.prepare(batch_size, n_epoch - begin_epoch)

        
        
        self._get_valid_otf_data()
        self._get_test_data()
        self._make_summaries(sess)
        
        """Pre-train the resolver independently
        """
        if begin_epoch == 0 and self.pretrain:
            training_dataset.reset(1)
            while training_dataset.hasNext():
                step_time = time.time()
                HR_batch, LR_batch, MR_batch, cursor, epoch = training_dataset.iter()
                
                evaluated = sess.run(self.pretrain_op, {self.plchdr_lr : LR_batch, self.plchdr_mr : MR_batch})         
                
                print("Pretrain iter:[%d/%d] times: %4.3fs" % (cursor + 1, n_training_pairs, time.time() - step_time))
                if verbose:
                    losses_val = {name : value for name, value in evaluated.items() if 'loss' in name}
                    print(losses_val)

        else:
            begin_epoch = self._load_designated_ckpt(begin_epoch, sess)   
            
        """
        training
        """
        fetches = dict(self.loss, **(self.optim))
        fetches['batch_summary'] = self.summary_op_training

        training_dataset.reset(n_epoch)
        while training_dataset.hasNext():
            step_time = time.time()
            HR_batch, LR_batch, MR_batch, cursor, epoch = training_dataset.iter()

            epoch += begin_epoch

            if (cursor == 0):
                print('')
                self._record_avg_test_loss(epoch, sess)

            if (self.archi1 is not None): 
                evaluated = sess.run(fetches, {self.plchdr_lr : LR_batch, self.plchdr_hr : HR_batch, self.plchdr_mr : MR_batch})
            else:
                evaluated = sess.run(fetches, {self.plchdr_lr : LR_batch, self.plchdr_hr : HR_batch})
            print("\rEpoch:[%d/%d] iter:[%d/%d] times: %4.3fs     " % (epoch, n_epoch, cursor + 1, n_training_pairs, time.time() - step_time), end='')
            
            if verbose:
                losses_val = {name : value for name, value in evaluated.items() if 'loss' in name}
                print(losses_val, end='')

            n_iters_passed = epoch * (n_training_pairs // batch_size) + cursor / batch_size
            self.training_loss_writer.add_summary(evaluated['batch_summary'], n_iters_passed)

            if (epoch != 0 and epoch % decay_every == 0 and cursor == n_training_pairs - 1 ):
                self._adjust_learning_rate(epoch, sess)
               
            if (epoch % ckpt_saving_interval == 0) and (cursor == 0):
                self._save_intermediate_ckpt(epoch, sess)
                if test:
                    self._add_test_summary(n_iters_passed, epoch, sess)

            

class Trainer(BaseTrainer):
    """Trainer for common one-stage net
    """
    def __init__(self, dataset, architecture='RDN', visualize_features=False):
        super(Trainer, self).__init__(dataset, None, architecture, visualize_features)
    def build_graph(self):
        super(Trainer, self).build_graph()
        self.archi = self.archi2

        with tf.device('/gpu:1'):
            
            variable_tag = '1stage_%s' % self.archi
            if self.archi is 'rdn':
                net = res_dense_net(self.plchdr_lr, factor=config.factor, reuse=False, bn=using_batch_norm, name=variable_tag)
                net_test = res_dense_net(self.plchdr_lr, factor=config.factor, reuse=True, bn=using_batch_norm, name=variable_tag)   

            elif self.archi is 'unet':
                self.plchdr_lr = tf.placeholder("float", [batch_size] + hr_size, name="LR")    
                net = unet3d(self.plchdr_lr, upscale=False, reuse=False, is_train=True, name=variable_tag)
                net_test = unet3d(self.plchdr_lr, upscale=False, reuse=True, is_train=False, name=variable_tag)
            elif self.archi is 'dbpn':
                net = DBPN(self.plchdr_lr, upscale=True, factor=config.factor, reuse=False, name=variable_tag)
                net_test = DBPN(self.plchdr_lr, upscale=True, factor=config.factor, reuse=True, name=variable_tag)
            else:
                raise Exception('unknow architecture: %s' % self.archi)

            #net = DBPN(self.plchdr_lr, upscale=True, reuse=False, name=variable_tag)

        #net_test = DBPN(self.plchdr_lr, upscale=True, reuse=True, name=variable_tag) 
        

        self.net = net
        net_vars = tl.layers.get_variables_with_name(variable_tag, train_only=True, printable=False)

        ln_loss = loss_fn(self.plchdr_hr, net.outputs)
        ln_loss_test = loss_fn(self.plchdr_hr, net_test.outputs)
        ln_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(ln_loss, var_list=net_vars)

        
        self.loss.update({'ln_loss' : ln_loss})
        self.loss_test.update({'ln_loss_test' : ln_loss_test})
        self.optim.update({'ln_optim' : ln_optim})

        if using_edge_loss:
            loss_edges = edges_loss(net.outputs, self.plchdr_hr)
            e_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_edges, var_list=net_vars)
            self.loss.update({'edge_loss' : loss_edges})
            self.optim.update({'e_optim' : e_optim})
        if using_grad_loss:
            loss_grad = img_gradient_loss(net.outputs, self.plchdr_hr)
            g_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_grad, var_list=net_vars)
            self.loss.update({'grad_loss' : loss_grad})
            self.optim.update({'g_optim' : g_optim})

class SegmentedTrainer(BaseTrainer):
    """Trainer for two-stage net 
    """
    def __init__(self, dataset, archi1, archi2, visualize_features=False):
        super(SegmentedTrainer, self).__init__(dataset, archi1, archi2, visualize_features)

    def build_graph(self):  
        super(SegmentedTrainer, self).build_graph()
        variable_tag_res = 'Resolve'
        variable_tag_interp = 'Interp'

        # if self.archi1 == 'dbpn':
        #     net1 = DBPN 
        # elif self.archi1 == 'denoise'
        #     net1 = denoise_net
        # else:
        #     _raise(ValueError())   

        var_tag_n2 = variable_tag_interp
        self.plchdr_mr = tf.placeholder("float", [batch_size] + lr_size, name="MR")  
        with tf.device('/gpu:%d' % 1):
            if self.archi1 == 'dbpn':
                net_stage1      = DBPN(self.plchdr_lr, upscale=False, name=variable_tag_res)
                net_stage1_test = DBPN(self.plchdr_lr, upscale=False, reuse=True, name=variable_tag_res)
            elif self.archi1 == 'denoise':
                net_stage1      = denoise_net(self.plchdr_lr, reuse=False, name=variable_tag_res)
                net_stage1_test = denoise_net(self.plchdr_lr, reuse=True, name=variable_tag_res)
            else:
                _raise(ValueError())   

        with tf.device('/gpu:%d' % 2):
            if self.archi2 == 'rdn':
                net_stage2      = res_dense_net(net_stage1.outputs, factor=config.factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=True, name=variable_tag_interp)
                net_stage2_test = res_dense_net(net_stage1_test.outputs, factor=config.factor, conv_kernel=conv_kernel, bn=using_batch_norm, reuse=True, is_train=False, name=variable_tag_interp)
            else:
                _raise(ValueError())   

        self.resolver = net_stage1
        self.interpolator = net_stage2

            
        net_stage1.print_params(details=False)
        net_stage2.print_params(details=False)

        #vars_n1 = tl.layers.get_variables_with_name(variable_tag_res, train_only=True, printable=False)
        vars_n2 = tl.layers.get_variables_with_name(var_tag_n2, train_only=True, printable=False)
        
        loss_training_n1 = loss_fn(self.plchdr_mr, net_stage1.outputs)
        loss_training_n2 = loss_fn(self.plchdr_hr, net_stage2.outputs)
        
        loss_test_n1 = loss_fn(self.plchdr_mr, net_stage1_test.outputs)
        loss_test_n2 = loss_fn(self.plchdr_hr, net_stage2_test.outputs)

        loss_training = loss_training_n1 + loss_training_n2
        loss_test = loss_test_n2 + loss_test_n1

        #n1_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_training, var_list=vars_n1)
        #n2_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_training_n2, var_list=vars_n2)
        #n1_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_training_n2)
        n1_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_training_n1)
        n_optim  = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_training)
        
        if self.pretrain:
            self.pretrain_op = {}
            self.pretrain_op.update({'loss_pretrain' : loss_training_n1, 'optim_pretrain' : n1_optim})

        self.loss.update({'loss_training' : loss_training, 'loss_training_n2' : loss_training_n2, 'loss_training_n1' : loss_training_n1})
        self.loss_test.update({'loss_test' : loss_test, 'loss_test_n2' : loss_test_n2, 'loss_test_n1' : loss_test_n1})
        #self.optim.update({'n1_optim' : n1_optim, 'n2_optim' : n2_optim, 'n_optim' : n_optim})
        self.optim.update({'n_optim' : n_optim})

        if using_edge_loss:
            loss_edges = edges_loss(net_stage2.outputs, self.plchdr_hr)
            e_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_edges, var_list=vars_n2)
            self.loss.update({'edge_loss' : loss_edges})
            self.optim.update({'e_optim' : e_optim})

        if using_grad_loss:
            loss_grad = img_gradient_loss(net_stage2.outputs, self.plchdr_hr)
            g_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_grad, var_list=vars_n2)
            self.loss.update({'grad_loss' : loss_grad})
            self.optim.update({'g_optim' : g_optim})

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

    def _get_test_data(self):
        super(SegmentedTrainer, self)._get_test_data()        
        write3d(self.test_mr[0 : batch_size], test_saving_dir+'test_mr.tif')

class FusedSegTrainer(BaseTrainer):
    """Trainer for a fused 2-stage net, used after SegmentedTrainer
    """
    def __init__(self, dataset, architecture='fused', visualize_features=False):
        super(FusedSegTrainer, self).__init__(dataset, architecture, visualize_features)
    def build_graph(self):
        super(FusedSegTrainer, self).build_graph()

        with tf.device('/gpu:1'):
            
            variable_tag = self.archi
            net, resolver, removed = fusedSegNet(self.plchdr_lr, factor=config.factor, reuse=False, name=variable_tag)
            net_test, _, _ = fusedSegNet(self.plchdr_lr, factor=config.factor, reuse=True, name=variable_tag)


        self.net, self.resolver, self.removed = net, resolver, removed
        net_vars = tl.layers.get_variables_with_name(variable_tag, train_only=True, printable=False)

        ln_loss = loss_fn(self.plchdr_hr, net.outputs)
        ln_loss_test = loss_fn(self.plchdr_hr, net_test.outputs)
        ln_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(ln_loss, var_list=net_vars)

        
        self.loss.update({'ln_loss' : ln_loss})
        self.loss_test.update({'ln_loss_test' : ln_loss_test})
        self.optim.update({'ln_optim' : ln_optim})

        if using_edge_loss:
            loss_edges = edges_loss(net.outputs, self.plchdr_hr)
            e_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_edges, var_list=net_vars)
            self.loss.update({'edge_loss' : loss_edges})
            self.optim.update({'e_optim' : e_optim})
        if using_grad_loss:
            loss_grad = img_gradient_loss(net.outputs, self.plchdr_hr)
            g_optim = tf.train.AdamOptimizer(self.learning_rate_var, beta1=beta1).minimize(loss_grad, var_list=net_vars)
            self.loss.update({'grad_loss' : loss_grad})
            self.optim.update({'g_optim' : g_optim})

    def _load_designated_ckpt(self, begin_epoch, sess):
        print("="*66)
        if (begin_epoch != 0):
            resolve_ckpt = self._traversal_through_ckpts(checkpoint_dir, begin_epoch, 'resolve')
            interp_ckpt  = self._traversal_through_ckpts(checkpoint_dir,begin_epoch, 'interp')
            
            if interp_ckpt is None: 
                raise Exception('designated checkpoint file for interpolator not found')
            if resolve_ckpt is None: 
                raise Exception('designated checkpoint file for resolver not found')
            load_ckpt_partial(checkpoint_dir+'/{}'.format(resolve_ckpt), self.resolver, 0, self.removed['dbpn'], sess)                
            load_ckpt_partial(checkpoint_dir+'/{}'.format(interp_ckpt), self.net, 56, self.removed['rdn'], sess)  
            
            return begin_epoch
        else:
            #return self._find_available_ckpt(n_epoch, sess)
            return 0
    


   
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    args = parser.parse_args()
    begin_epoch = args.ckpt

    train_lr_path = config.TRAIN.lr_img_path
    train_hr_path = config.TRAIN.hr_img_path
    train_mr_path = config.TRAIN.mr_img_path if config.archi1 is not None else None
    valid_lr_path = config.TRAIN.valid_lr_path  # lr measuremnet for validation during the training   

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

    transform = [interpolate3d, None] if config.archi2 == 'unet' else None
    lr_size   = hr_size if config.archi2 == 'unet' else lr_size
    dataset = Dataset(
        lr_size       =lr_size, 
        hr_size       =hr_size, 
        train_lr_path = train_lr_path,
        train_hr_path = train_hr_path,
        test_lr_path  =test_lr_path,  
        test_hr_path  =test_hr_path,
        mr_size       =mr_size,
        train_mr_path =train_mr_path,
        test_mr_path  =test_mr_path,
        valid_lr_path =valid_lr_path,
        dtype         =np.float32,
        transforms    = transform, 
        factor        =config.factor
    )

    if config.archi1 is not None:
        # trainer = FusedSegTrainer(dataset, architecture=config.archi)
        trainer = SegmentedTrainer(dataset, archi1=config.archi1, archi2=config.archi2)
    else:
        trainer = Trainer(dataset, architecture=config.archi2)
        
    trainer.build_graph()
    trainer.train(begin_epoch=begin_epoch, test=True, verbose=True)

    
                  
