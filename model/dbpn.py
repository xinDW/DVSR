import tensorflow as tf 
import tensorlayer as tl

from tensorlayer.layers import InputLayer
from .custom import conv3d, conv2d, conv3d_transpose, conv2d_transpose, concat, prelu

__all__ = ['DBPN', 'DBPN_front']

def _raise(e):
    raise(e)

def _get_conv_fn(conv_type):
    conv_type in ['3d', '2d'] or _raise(ValueError('conv_type must be in ["3d", "2d"] but given is %s' % conv_type))
    [conv, deconv] = [conv3d, conv3d_transpose] if conv_type is '3d' else [conv2d, conv2d_transpose]
    return conv, deconv

def _up_block(x, n_filters, k_size=8, stride=4, act=tf.identity, conv_type='3d', name='_up_block'):
    '''
    up-scale the input x by a factor = stride
    '''
    conv, deconv = _get_conv_fn(conv_type)
    with tf.variable_scope(name):
        h0 = deconv(x, out_channels = n_filters, filter_size=k_size, stride=stride, act=act, name='deconv1')
        l0 = conv(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='conv')

        l0.outputs = l0.outputs - x.outputs
        h1 = deconv(l0, out_channels = n_filters, filter_size=k_size, stride=stride, act=act, name='deconv2')

        h1.outputs = h1.outputs + h0.outputs

        return h1

def _down_block(x, n_filters, k_size=8, stride=4, act=tf.identity, conv_type='3d', name='_down_block'):
    '''
    down-scale the input x by a factor = stride
    '''
    conv, deconv = _get_conv_fn(conv_type)
    with tf.variable_scope(name):
        l0 = conv(x, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='conv1')
        h0 = deconv(l0, out_channels = n_filters, filter_size=k_size, stride=stride, act=act, name='deconv')

        h0.outputs = h0.outputs - x.outputs
        l1 = conv(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='conv2')

        l1.outputs = l1.outputs + l0.outputs
        return l1

def _d_up_block(x, n_filters, k_size=8, stride=4, act=tf.identity, conv_type='3d', name='_d_up_block'):
    '''
    up-scale the input x by a factor = stride
    '''
    conv, deconv = _get_conv_fn(conv_type)
    with tf.variable_scope(name):
        x = conv(x, out_channels=n_filters, filter_size=1, stride=1, act=act, name='conv')
        h0 = deconv(x, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='up_conv1')
        l0 = conv(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='up_conv2')

        l0.outputs = l0.outputs - x.outputs
        h1 = deconv(l0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='up_conv3')

        h1.outputs = h1.outputs + h0.outputs
        return h1

def _d_down_block(x, n_filters, k_size=8, stride=4, act=tf.identity, conv_type='3d', name='_d_down_block'):
    conv, deconv = _get_conv_fn(conv_type)
    with tf.variable_scope(name):
        x = conv(x, out_channels=n_filters, filter_size=1, stride=1, act=act, name='conv')
        l0 = conv(x, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='down_conv1')
        h0 = deconv(l0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='down_conv2')

        h0.outputs = h0.outputs - x.outputs
        l1 = conv(h0, out_channels=n_filters, filter_size=k_size, stride=stride, act=act, name='down_conv3')

        l1.outputs = l1.outputs + l0.outputs
        return l1

def DBPN(input, conv_type='3d', feat=64, base_filter=32, upscale=False, factor=2, reuse=False, name='dbpn'):
    '''
    Dense-deep Back-projection Net
    Params:
        -conv_type   : in ['3d', '2d'], convolutional layer type
        -upscale: if False, the output will have the same size as the input LR, 
                else the output_size = 4 * input_size
    '''
    conv, _        = _get_conv_fn(conv_type)
    act            = prelu
    kernel         = 3
    stride         = factor if upscale else 2
    additional_up_down_pair = 1 if upscale else 2
    
    with tf.variable_scope(name, reuse=reuse):
        n_channels = input.shape[-1]
        x          = InputLayer(input, name='input')

        # initial feature extration
        x = conv(x, out_channels=feat, filter_size=3, act=act, name='feat0')
        x = conv(x, out_channels=base_filter, filter_size=1, act=act, name='feat1')

        #back-projection
        h1 = _up_block(x, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='up1')
        l1 = _down_block(h1, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='down1')
        h2 = _up_block(l1, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='up2')

        concat_h = concat([h2, h1])
        l = _d_down_block(concat_h, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='down2')

        concat_l = concat([l, l1])
        h = _d_up_block(concat_l, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='up3')

        for i in range(0, additional_up_down_pair):
            concat_h = concat([h, concat_h])
            l =  _d_down_block(concat_h, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='down%d' % (i + 3))

            concat_l = concat([l, concat_l])
            h = _d_up_block(concat_l, n_filters=base_filter, k_size=kernel, stride=stride, act=act, conv_type=conv_type, name='up%d' % (i + 4))

        concat_h = concat([h, concat_h])

        if upscale:
            x = conv(concat_h, out_channels=n_channels, filter_size=3, act=tf.tanh, name='output_conv')
        else:
            x = _down_block(concat_h, n_filters=1, k_size=3, stride=stride, conv_type=conv_type, name='out')
        return x

def DBPN_front(input, feat=64, base_filter=32, upscale=False, reuse=False, name='dbpn'):
    '''
    DBPN with last several layers removed
    out_size = in_size * 2
    Params:
        
    '''
    act = prelu
    kernel = 3

    stride = 2
    additional_up_down_pair = 2
    
    with tf.variable_scope(name, reuse=reuse):
        n_channels = input.shape[-1]
        x = InputLayer(input, name='input')

        # initial feature extration
        x = conv3d(x, out_channels=feat, filter_size=3, act=act, name='feat0')
        x = conv3d(x, out_channels=base_filter, filter_size=1, act=act, name='feat1')

        #back-projection
        h1 = _up_block(x, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up1')
        l1 = _down_block(h1, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='down1')
        h2 = _up_block(l1, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up2')

        concat_h = concat([h2, h1])
        l = _d_down_block(concat_h, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='down2')

        concat_l = concat([l, l1])
        h = _d_up_block(concat_l, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up3')

        for i in range(0, additional_up_down_pair - 1):
            concat_h = concat([h, concat_h])
            l =  _d_down_block(concat_h, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='down%d' % (i + 3))

            concat_l = concat([l, concat_l])
            h = _d_up_block(concat_l, n_filters=base_filter, k_size=kernel, stride=stride, act=act, name='up%d' % (i + 4))

        concat_h = concat([h, concat_h])

        return concat_h


