import tensorflow as tf
import tensorlayer as tl

from .custom import conv3d, batch_norm, concat, LReluLayer, ReluLayer, SubVoxelConv, max_pool3d

__all__ = ['unet3d']

def upconv3d(layer, out_channels, factor=2, mode='subpixel', output_shape=None, act=tf.identity, name='upconv'):  
    with tf.variable_scope(name):
        if mode == 'subpixel':
            in_channels = layer.outputs.shape.as_list()[-1]
            n = conv3d(layer, out_channels*(factor**3), 1, 1, act=tf.identity)
            n = SubVoxelConv(n, scale=factor, n_out_channel=None, act=act, name=mode)
            return n
            
        elif mode == 'deconv':
            batch, depth, height, width, in_channels = layer.outputs.shape.as_list()
            
            if output_shape is None:
                output_shape = (batch, depth*factor, height*factor, width*factor, out_channels)
            else:
                if len(output_shape) == 3:
                    output_shape = [batch] + output_shape + [out_channels]

            n = tl.layers.DeConv3dLayer(layer, act=act, 
            shape=(1, 1, 1, out_channels, in_channels), 
            output_shape=output_shape, 
            strides=(1, factor, factor, factor, 1), padding='SAME', W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0), name=mode)
            
            return n
            
        else:
            raise Exception('unknown mode : %s' % mode)
        
def unet3d(LR, ngf=64, upscale=False, reuse=False, is_train=False, name='unet3d'):
    
    '''
    Params:
        LR - [batch, depth, height, width, channels]
    '''
    f_size = 3
    act = tf.nn.leaky_relu
    layers = []
    with tf.variable_scope(name, reuse=reuse):
        n = tl.layers.InputLayer(LR, name='lr_input')
        n = conv3d(n, ngf, f_size, 1, act=act, name='conv0')
        layers.append(n)
        
        layer_specs = [
            ngf * 2, 
            ngf * 4,
            ngf * 8,
            #ngf * 16,
            #ngf * 8,
            #ngf * 8, 
            #ngf * 8
        ]
        
        for out_channels in layer_specs:
            with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
                #rect = LReluLayer(layers[-1], alpha=0.2)
                conv = conv3d(layers[-1], out_channels, f_size, 1, act=act)
                pool = max_pool3d(conv, filter_size=f_size, stride=2, padding='SAME', name='maxpool')
                layers.append(pool)
                print(pool.outputs.shape)
        
        layer_specs.reverse()
        layer_specs = [l // 2 for l in layer_specs]
        
        encoder_layers_num = len(layers)
        for decoder_layer, out_channels in enumerate(layer_specs):
            skip_layer = encoder_layers_num - decoder_layer - 1
            _, d, h, w, _ = layers[skip_layer - 1].outputs.shape.as_list()
            print([d, h, w])
            with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = layers[-1]
                else:
                    input = concat([layers[-1], layers[skip_layer]])
                rect = ReluLayer(input)
                out = upconv3d(rect, out_channels, mode='deconv', output_shape=[d, h, w])
                #out = batch_norm(out, is_train=is_train)
                layers.append(out)
        
        with tf.variable_scope('out'):
            n = concat([layers[-1], layers[0]])

            if upscale is False:
                output = conv3d(n, 1, f_size, 1, act=tf.nn.tanh, name='conv')
            else:
                output = upconv3d(n, out_channels=8, name='upsampling1')
                output = upconv3d(output, out_channels=1, act=tf.tanh, name='upsampling2')
            
            return output
        