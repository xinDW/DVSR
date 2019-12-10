import tensorflow as tf
import tensorlayer as tl

from .custom import conv3d, conv3d_transpose, batch_norm, concat, upscale
from tensorlayer.layers import Layer, InputLayer, ElementwiseLayer

def denoise_net (lr, reuse=False, name='RDN'):
    conv_kernel = 3
    act         = tf.nn.relu

    def conv_block(input, act, name):
        with tf.variable_scope(name):
            n = input if isinstance(input, Layer) else InputLayer(input, 'n0')
            channels_num = n.outputs.shape[-1]
            n = conv3d(n, out_channels=channels_num, filter_size=1, act=act, name='n1')
            n = conv3d(n, out_channels=channels_num // 4, filter_size=3, act=act, name='n2')
            n = conv3d(n, out_channels=channels_num, filter_size=1, act=act, name='n3')
        return n

    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(lr, 'lr') if not isinstance(lr, Layer) else lr
    
        n = conv3d(n, out_channels=64, filter_size=conv_kernel, act=act, name='conv1')
        for i in range(3):       
            n = conv_block(n, act=act, name='conv_block%d' % i)
        n = conv3d(n, out_channels=1, filter_size=3, act=tf.identity, name='out')

    return n