import tensorflow as tf 
import tensorlayer as tl

from tensorlayer.layers import InputLayer, ConcatLayer, ElementwiseLayer, PReluLayer
from model import conv3d, conv3d_transpose, concat

def prelu(x, name='prelu'):
    w_shape = x.get_shape()[-1]
    with tf.variable_scope(name):
        alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=tf.constant_initializer(value=0.0) )
        out = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
        return out

def up_block(x, n_filters, k_size=8, stride=4, act=tf.identity, name='up_block'):
    '''
    up-scale the input x by a factor = stride
    '''
    with tf.variable_scope(name):
        h0 = conv3d_transpose(x, out_channels = n_filters, filter_size=k_size, stride=stride, act=act, name='deconv1')
        l0 = conv3d(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='conv')

        l0.outputs = l0.outputs - x.outputs
        h1 = conv3d_transpose(l0, out_channels = n_filters, filter_size=k_size, stride=stride, act=act, name='deconv2')

        h1.outputs = h1.outputs + h0.outputs

        return h1

def down_block(x, n_filters, k_size=8, stride=4, act=tf.identity, name='down_block'):
    '''
    down-scale the input x by a factor = stride
    '''
    with tf.variable_scope(name):
        l0 = conv3d(x, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='conv1')
        h0 = conv3d_transpose(l0, out_channels = n_filters, filter_size=k_size, stride=stride, act=act, name='deconv')

        h0.outputs = h0.outputs - x.outputs
        l1 = conv3d(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='conv2')

        l1.outputs = l1.outputs + l0.outputs
        return l1

def d_up_block(x, n_filters, k_size=8, stride=4, act=tf.identity, name='d_up_block'):
    '''
    up-scale the input x by a factor = stride
    '''
    with tf.variable_scope(name):
        x = conv3d(x, out_channels=n_filters, filter_size=1, stride=1, act=act, name='conv')
        h0 = conv3d_transpose(x, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='up_conv1')
        l0 = conv3d(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='up_conv2')

        l0.outputs = l0.outputs - x.outputs
        h1 = conv3d_transpose(l0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='up_conv3')

        h1.outputs = h1.outputs + h0.outputs
        return h1

def d_down_block(x, n_filters, k_size=8, stride=4, act=tf.identity, name='d_down_block'):
    with tf.variable_scope(name):
        x = conv3d(x, out_channels=n_filters, filter_size=1, stride=1, act=act, name='conv')
        l0 = conv3d(x, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='down_conv1')
        h0 = conv3d_transpose(l0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='down_conv2')

        h0.outputs = h0.outputs - x.outputs
        l1 = conv3d(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='down_conv3')

        l1.outputs = l1.outputs + l0.outputs
        return l1

def DBPN(input, feat=64, base_filter=32, name='dbpn'):
    '''
    Dense-deep Back-projection Net
    '''
    act = prelu
    kernel = 3
    stride = 2

    with tf.variable_scope(name):
        n_channels = input.shape[-1]
        x = InputLayer(input, name='input')

        # initial feature extration
        x = conv3d(x, out_channels=feat, filter_size=3, act=act, name='feat0')
        x = conv3d(x, out_channels=base_filter, filter_size=1, act=act, name='feat1')

        #back-projection
        h1 = up_block(x, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up1')
        l1 = down_block(h1, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='down1')
        h2 = up_block(l1, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up2')

        concat_h = concat([h2, h1])
        l = d_down_block(concat_h, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='down2')

        concat_l = concat([l, l1])
        h = d_up_block(concat_l, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up3')

        for i in range(0, 2):
            concat_h = concat([h, concat_h])
            l =  d_down_block(concat_h, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='down%d' % (i + 3))

            concat_l = concat([l, concat_l])
            h = d_up_block(concat_l, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up%d' % (i + 4))

        concat_h = concat([h, concat_h])
        #x =  conv3d(concat_h, out_channels=n_channels, filter_size=3, name='output_conv')
        x = down_block(concat_h, n_filters=1, k_size=3, stride=stride, name='out')
        return x
