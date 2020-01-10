import tensorflow as tf
import tensorlayer as tl

from .custom import conv3d, conv3d_transpose, batch_norm, concat, upscale
from tensorlayer.layers import Layer, InputLayer, ElementwiseLayer

__all__ = ['res_dense_net', 'RDN_end']
   
def res_dense_block(preceding, G=64, conv_kernel=3, bn=False, is_train=True, name='rdb'):
    """
    Resifual dense block
    Params : 
        preceding - An Layer class, feature maps of preceding block 
        G         - Growth rate of feature maps
    """
    G0 = preceding.outputs.shape[-1]
    if G0 != G:
        raise Exception('G0(%d) and G(%d) must be equal in RDB' % (G0, G))
    
    act = tf.nn.relu
    with tf.variable_scope(name):
        n1 = conv3d(preceding, out_channels=G, filter_size=conv_kernel, stride=1, act=act, name='conv1')
        if bn: n1 = batch_norm(n1, is_train=is_train, name='bn1') 
        n2 = concat([preceding, n1], name='conv2_in')
        n2 = conv3d(n2, out_channels=G, filter_size=conv_kernel, stride=1, act=act, name='conv2')
        if bn: n2 = batch_norm(n2, is_train=is_train, name='bn2') 
        n3 = concat([preceding, n1, n2], name='conv3_in')
        n3 = conv3d(n3, out_channels=G, filter_size=conv_kernel, stride=1, act=act, name='conv3')
        if bn: n3 = batch_norm(n3, is_train=is_train, name='bn3') 
        
        # local feature fusion (LFF)
        n4 = concat([preceding, n1, n2, n3], name='conv4_in')
        n4 = conv3d(n4, out_channels=G, filter_size=1, name='conv4')
        if bn: n4 = batch_norm(n4, is_train=is_train, name='bn4') 
        
        # local residual learning (LRL)
        out = ElementwiseLayer([preceding, n4], combine_fn=tf.add, name='out')
        
        return out



def res_dense_net(lr, factor=4, conv_kernel=3, bn=False, is_train=True, format_out=True, reuse=False, name='RDN'):
    '''Residual Dense net
    Params:
      -factor: super-resolution enhancement factor 
      -reuse: reuse the variables or not (in tf.variable_scope(name))
      -bn: whether use batch norm after conv layers or not
      -is_train: paramete with the identical name in tl.layer.BatchNormLayer (only valid when 'bn' == True)
      -format_out: if False, keep the increased pixels in channels dimension. Else re-arrange them into spatial dimensions(what the SubvoxelConv does exactly)
    '''
    assert factor in [1, 2, 3, 4]

    G0 = 64
    with tf.variable_scope(name, reuse=reuse):


      n = InputLayer(lr, 'lr') if not isinstance(lr, Layer) else lr
      
      # shallow feature extraction layers
      n1 = conv3d(n, out_channels=G0, filter_size=conv_kernel, name='shallow1')
      if bn: n1 = batch_norm(n1, is_train=is_train, name='bn1') 
      n2 = conv3d(n1, out_channels=G0, filter_size=conv_kernel, name='shallow2')
      if bn: n2 = batch_norm(n2, is_train=is_train, name='bn2') 
      
      n3 = res_dense_block(n2, conv_kernel=conv_kernel, bn=bn, name='rdb1')
      n4 = res_dense_block(n3, conv_kernel=conv_kernel, bn=bn, name='rdb2')
      n5 = res_dense_block(n4, conv_kernel=conv_kernel, bn=bn, name='rdb3')

      # global feature fusion (GFF)
      n6 = concat([n3, n4, n5], name='gff')
      n6 = conv3d(n6, out_channels=G0, filter_size=1, name='gff/conv1')
      if bn: n6 = batch_norm(n6, is_train=is_train, name='bn3') 
      n6 = conv3d(n6, out_channels=G0, filter_size=conv_kernel, name='gff/conv2')
      if bn: n6 = batch_norm(n6, is_train=is_train, name='bn4') 
      
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
        else :
          n8 = n7
        # out = conv3d(n8, out_channels=1, filter_size=conv_kernel, act=tf.tanh, name='out')
        out = conv3d(n8, out_channels=1, filter_size=conv_kernel, name='out')
    
      else:
        out = n7

      return out        

def RDN_end(feat, factor=4, conv_kernel=3, bn=False, is_train=True, format_out=True, reuse=False, name='RDN'):
    '''Residual Dense net without the first 2 input layers
    Params:
      see res_dense_net
    '''
    assert factor in [2, 4]

    G0 = 64
    with tf.variable_scope(name, reuse=reuse):
      n = InputLayer(feat, 'lr') if not isinstance(feat, Layer) else feat
     
      # shallow feature extraction layers

      # n1 = conv3d(n, out_channels=G0, filter_size=conv_kernel, name='shallow1')
      # if bn: n1 = batch_norm(n1, is_train=is_train, name='bn1') 
      # n2 = conv3d(n1, out_channels=G0, filter_size=conv_kernel, name='shallow2')
      # if bn: n2 = batch_norm(n2, is_train=is_train, name='bn2') 
      
      n3 = res_dense_block(n, conv_kernel=conv_kernel, bn=bn, name='rdb1')
      n4 = res_dense_block(n3, conv_kernel=conv_kernel, bn=bn, name='rdb2')
      n5 = res_dense_block(n4, conv_kernel=conv_kernel, bn=bn, name='rdb3')

      # global feature fusion (GFF)
      n6 = concat([n3, n4, n5], name='gff')
      n6 = conv3d(n6, out_channels=G0, filter_size=1, name='gff/conv1')
      if bn: n6 = batch_norm(n6, is_train=is_train, name='bn3') 
      n6 = conv3d(n6, out_channels=G0, filter_size=conv_kernel, name='gff/conv2')
      if bn: n6 = batch_norm(n6, is_train=is_train, name='bn4') 
      
      # global residual learning 
      #n7 = ElementwiseLayer([n6, n1], combine_fn=tf.add, name='grl')
      n7 = n6
      
      if factor == 4:
        n8 = upscale(n7, scale=2, name='upscale1')
      else :
        n8 = n7
      #out = conv3d(n8, out_channels=1, filter_size=conv_kernel, act=tf.sigmoid, name='out')
      out = conv3d(n8, out_channels=1, filter_size=conv_kernel, name='out')
    
      

      return out        