import tensorflow as tf
import tensorlayer as tl
import numpy as np

from .custom import conv3d, batch_norm, concat, add, LReluLayer, ReluLayer, SubVoxelConv, max_pool3d

__all__ = ['unet_care']

def conv_block(layer, n_filter, kernel_size,
                activation=tf.nn.relu,
                border_mode="SAME",
                # init="glorot_uniform",
                name='conv3d'):

    s = conv3d(layer, out_channels=n_filter, filter_size=kernel_size, stride=1, act=activation, padding=border_mode, name=name)
    return s

def upsampling(layer, factor=2, output_shape=None, act=tf.identity, name='upsampling3d'):  
    with tf.variable_scope(name):
        n = layer
        batch, depth, height, width, in_channels = n.outputs.shape.as_list()
        
        if output_shape is None:
            output_shape = (batch, depth*factor, height*factor, width*factor, in_channels)
        else:
            if len(output_shape) == 3:
                output_shape = [batch] + output_shape + [in_channels]

        filter = tf.constant(np.ones([factor, factor, factor, in_channels, in_channels], np.float32))
        strides = [1, factor, factor, factor, 1]

        deconv = tf.nn.conv3d_transpose(value=n.outputs, 
            filter=filter, 
            output_shape=output_shape, 
            strides=strides,
            padding='SAME')
        n.outputs = deconv
        return n

def pooling(layer, pool=2, name='pooling3d'):
    return max_pool3d(layer, filter_size=2, stride=pool, name=name)

def unet_block(input, n_depth=2, n_filter_base=16, kernel_size=3, n_conv_per_depth=2,
               activation=tf.nn.relu,
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=2,
               kernel_init="glorot_uniform",
               name='unet_block'):

    
    if last_activation is None:
        last_activation = activation

    skip_layers = []

    with tf.variable_scope(name):
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(layer, n_filter_base * 2 ** n, kernel_size,
                                    activation=activation,
                                    name="down_level_%s_no_%s" % (n, i))

            skip_layers.append(layer)
            layer = pooling(layer, pool, name="max_%s" % n)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(layer, n_filter_base * 2 ** n_depth, kernel_size,
                                activation=activation,
                                name="middle_%s" % i)

        layer = conv_block(layer, n_filter_base * 2 ** max(0, n_depth - 1), kernel_size,
                            activation=activation,
                            name="middle_%s" % n_conv_per_depth)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = concat([upsampling(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(layer, n_filter_base * 2 ** n, kernel_size,
                                    activation=activation,
                                    name="up_level_%s_no_%s" % (n, i))

            layer = conv_block(layer, n_filter_base * 2 ** max(0, n - 1), kernel_size,
                                activation=activation if n > 0 else last_activation,
                                name="up_level_%s_no_%s" % (n, n_conv_per_depth))

        return layer


def unet_care(LR,
                last_activation=tf.identity,
                n_depth=4,
                n_filter_base=32,
                kernel_size=3,
                n_conv_per_depth=2,
                activation=tf.nn.relu,
                batch_norm=False,
                dropout=0.0,
                pool_size=2,
                n_channel_out=1,
                residual=False,
                reuse=False,
                name="unet_care"):
    '''
    Params:
        LR - [batch, depth, height, width, channels]
    '''

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    with tf.variable_scope(name, reuse=reuse):
        input = tl.layers.InputLayer(LR, name='lr_input')

        unet = unet_block(input, n_depth, n_filter_base, kernel_size,
                        activation=activation, dropout=dropout, batch_norm=batch_norm,
                        n_conv_per_depth=n_conv_per_depth, pool=pool_size)

        final = conv3d(unet, out_channels=n_channel_out, filter_size=1)
        if residual:
            final = add([final, input])
        final.outputs = last_activation(final.outputs)

        return final      