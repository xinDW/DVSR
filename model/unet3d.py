import tensorflow as tf
import tensorlayer as tl

from .custom import conv3d, batch_norm, concat, LReluLayer, ReluLayer, SubVoxelConv

__all__ = ['unet3d']

def upconv3d(layer, out_channels, factor=2, mode='subpixel', act=tf.identity, name='upconv'):  
    with tf.variable_scope(name):
        if mode == 'subpixel':
            in_channels = layer.outputs.shape.as_list()[-1]
            n = conv3d(layer, out_channels*(factor**3), 1, 1, act=tf.identity)
            n = SubVoxelConv(n, scale=factor, n_out_channel=None, act=act, name=mode)
            return n
            
        elif mode == 'deconv':
            batch, depth, height, width, in_channels = layer.outputs.shape.as_list()
            
            n = tl.layers.DeConv3dLayer(layer, act=act, 
            shape=(1, 1, 1, out_channels, in_channels), 
            output_shape=(batch, depth*factor, height*factor, width*factor, out_channels), 
            strides=(1, 2, 2, 2, 1), padding='SAME', W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0), name=mode)
            
            return n
            
        else:
            raise Exception('unknown mode : %s' % mode)
        
def unet3d(LR, ngf=64, reuse=False, is_train=False, name='unet3d'):
    '''
    Params:
        LR - [batch, depth, height, width, channels]
    '''
    in_channels = LR.shape.as_list()[-1]
    layers = []
    with tf.variable_scope(name, reuse=reuse):
        n = tl.layers.InputLayer(LR, name='lr_input')
        n = conv3d(n, ngf, 4, 1, name='conv0')
        #n = upconv3d(n, out_channels=ngf, name='upsampling1')
        layers.append(n)
        
        layer_specs = [
            ngf * 2, 
            ngf * 4,
            ngf * 8,
            ngf * 8,
            #ngf * 8,
            #ngf * 8, 
            #ngf * 8
        ]
        
        for out_channels in layer_specs:
            with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
                rect = LReluLayer(layers[-1], alpha=0.2)
                in_channels = rect.outputs.shape.as_list()[-1]
                conv = conv3d(rect, out_channels, 4, 2)
                #out = batch_norm(conv, is_train=is_train)
                layers.append(conv)
        
        layer_specs = [
            #ngf * 8, 
            #ngf * 8, 
            #ngf * 8,
            ngf * 8, 
            ngf * 4,
            ngf * 2,
            ngf
        ]
        
        encoder_layers_num = len(layers)
        for decoder_layer, out_channels in enumerate(layer_specs):
            skip_layer = encoder_layers_num - decoder_layer - 1
            with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = layers[-1]
                else:
                    input = concat([layers[-1], layers[skip_layer]])
                rect = ReluLayer(input)
                out = upconv3d(input, out_channels, mode='deconv')
                #out = batch_norm(out, is_train=is_train)
                layers.append(out)
        
        with tf.variable_scope('out'):
            input = concat([layers[-1], layers[0]])
            rect = ReluLayer(input)
            output = upconv3d(rect, out_channels=8, name='upsampling1')
            output = upconv3d(output, out_channels=1, act=tf.tanh, name='upsampling2')
            #rect.outputs = tf.tanh(rect.outputs)
            return output
        