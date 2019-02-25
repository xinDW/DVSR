import tensorflow as tf
import tensorlayer as tl

from utils import *
from config import *
from model import conv3d, conv3d_transpose
from custom import SubpixelConv3d
from tensorlayer.layers import InputLayer, ConcatLayer, ElementwiseLayer

conv_kernel = config.TRAIN.conv_kernel

def concat(layer, concat_dim=-1, name='concat'):
    return ConcatLayer(layer, concat_dim=concat_dim, name=name)
   
def res_dense_block(preceding, G=64, name='rdb'):
    """
    Resifual dense block
    Params : 
        preceding - An Layer class, feature maps of preceding block 
        G         - Growth rate of feature maps
    """
    G0 = preceding.outputs.shape[-1]
    if G0 != G:
        raise Exception('G0 and G must be equal in RDB')
    
    act = tf.nn.relu
    with tf.variable_scope(name):
        n1 = conv3d(preceding, out_channels=G, filter_size=conv_kernel, stride=1, act=act, name='conv1')
        
        n2 = concat([preceding, n1], name='conv2_in')
        n2 = conv3d(n2, out_channels=G, filter_size=conv_kernel, stride=1, act=act, name='conv2')
        
        n3 = concat([preceding, n1, n2], name='conv3_in')
        n3 = conv3d(n3, out_channels=G, filter_size=conv_kernel, stride=1, act=act, name='conv3')
        
        # local feature fusion (LFF)
        n4 = concat([preceding, n1, n2, n3], name='conv4_in')
        n4 = conv3d(n4, out_channels=G, filter_size=1, name='conv4')
        
        # local residual learning (LRL)
        out = ElementwiseLayer([preceding, n4], combine_fn=tf.add, name='out')
        
        return out

def upscale(layer, scale=2, name='upscale'):
    return SubpixelConv3d(layer, scale=scale, n_out_channel=None, act=tf.identity, name=name)
        
def res_dense_net(lr, factor=4, reuse=False, format_out=True, name='RDN'):
    G0 = 64
    with tf.variable_scope(name, reuse=reuse):
      n = InputLayer(lr, 'lr')
      
      # shallow feature extraction layers
      n1 = conv3d(n, out_channels=G0, filter_size=conv_kernel, name='shallow1')
      n2 = conv3d(n1, out_channels=G0, filter_size=conv_kernel, name='shallow2')
      
      n3 = res_dense_block(n2, name='rdb1')
      n4 = res_dense_block(n3, name='rdb2')
      n5 = res_dense_block(n4, name='rdb3')

      # global feature fusion (GFF)
      n6 = concat([n3, n4, n5], name='gff')
      n6 = conv3d(n6, out_channels=G0, filter_size=1, name='gff/conv1')
      n6 = conv3d(n6, out_channels=G0, filter_size=conv_kernel, name='gff/conv2')
      
      # global residual learning 
      n7 = ElementwiseLayer([n6, n1], combine_fn=tf.add, name='grl')
      
      if format_out:
        if factor == 4:
          n8 = upscale(n7, scale=2, name='upscale1')
          n8 = upscale(n8, scale=2, name='upscale2')
        elif factor == 3:
          n8 = conv3d(n7, out_channels=27, filter_size=3, name='conv3')
          n8 = upscale(n8, scale=3, name='upscale1')
        elif factor == 2:
          #n8 = conv3d(n7, out_channels=8, filter_size=3, name='conv3')
          n8 = upscale(n7, scale=2, name='upscale1')
          
        out = conv3d(n8, out_channels=1, filter_size=conv_kernel, act=tf.tanh, name='out')
    
      else:
        out = n7
      return out        
        

def res_dense_net_4gpu(lr, factor=4, format_out=True, name='generator'):
    G0 = 64
     
    with tf.variable_scope(name):
      with tf.variable_scope('RDN'):
        with tf.device('/gpu:0'):  
          n = InputLayer(lr, 'lr')
          
          # shallow feature extraction layers
          n1 = conv3d(n, out_channels=G0, filter_size=conv_kernel, name='shallow1')
          n2 = conv3d(n1, out_channels=G0, filter_size=conv_kernel, name='shallow2')
          
          with tf.device('/gpu:1'):
            n3 = res_dense_block(n2, name='rdb1')
          with tf.device('/gpu:2'):
            n4 = res_dense_block(n3, name='rdb2')
          with tf.device('/gpu:3'):
            n5 = res_dense_block(n4, name='rdb3')

          # global feature fusion (GFF)
          n6 = concat([n3, n4, n5], name='gff')
          n6 = conv3d(n6, out_channels=G0, filter_size=1, name='gff/conv1')
          n6 = conv3d(n6, out_channels=G0, filter_size=conv_kernel, name='gff/conv2')
          
          # global residual learning 
          n7 = ElementwiseLayer([n6, n1], combine_fn=tf.add, name='grl')
          
          if factor == 4:
            n8 = upscale(n7, scale=2, name='upscale1')
            n8 = upscale(n8, scale=2, name='upscale2')
          elif factor == 3:
            n8 = n6 = conv3d(n7, out_channels=27, filter_size=1, name='conv3')
            n8 = upscale(n8, scale=3, name='upscale1')
            
          out = conv3d(n8, out_channels=1, filter_size=conv_kernel, act=tf.tanh, name='out')
          
          return out        
        