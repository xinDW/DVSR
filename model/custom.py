import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Layer, Conv3dLayer, DeConv3dLayer, Conv2dLayer, DeConv2dLayer, ConcatLayer, BatchNormLayer, MaxPool3d

__all__ = ['conv3d',
    'conv3d_transpose',
    'upscale',
    'concat',
    'batch_norm',
    'max_pool3d',
    'prelu',
    'SubVoxelConv',
    'LReluLayer',
    'ReluLayer'
]

w_init = tf.random_normal_initializer(stddev=0.02)
#b_init = tf.constant_initializer(value=0.0)
b_init=None
g_init = tf.random_normal_initializer(1., 0.02)

def conv3d_transpose(input, out_channels, filter_size, stride, act=None, padding='SAME', name='conv3d_transpose' ):
    batch, depth, height, width, in_channels = input.outputs.get_shape().as_list()
    shape = [filter_size, filter_size, filter_size, out_channels, in_channels]
    output_shape = [batch, depth*stride, height*stride, width*stride, out_channels]
    strides = [1, stride, stride, stride, 1]
    return DeConv3dLayer(input, act=act, shape=shape, output_shape=output_shape, strides=strides, padding=padding, W_init=w_init, b_init=b_init, name=name)

def conv3d_transpose2(input, out_channels, filter_size, stride, padding='SAME', name='conv3d_transpose' ):
    return tf.layers.Conv3DTranspose(filters=out_channels, kernel_size=(filter_size, filter_size, filter_size), strides=(stride, stride, stride), padding=padding, kernel_initializer=w_init, bias_initializer=b_init, name=name);

def conv3d(layer, out_channels, filter_size=3, stride=1, act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv3d'):
    """
    Params
        shape - shape of the filters, [filter_depth, filter_height, filter_width, in_channels, out_channels].
        strides - shift in [batch in_depth in_height in_width in_channels] 
    """
    in_channels = layer.outputs.get_shape().as_list()[-1]
    shape=[filter_size,filter_size,filter_size,in_channels,out_channels]
    strides=[1,stride,stride,stride,1]
    return Conv3dLayer(layer, act=act, shape=shape, strides=strides, padding=padding, W_init=W_init, b_init=b_init, name=name)

def conv2d(layer, out_channels, filter_size=3, stride=1, act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv2d'):
    in_channels = layer.outputs.get_shape().as_list()[-1]
    return Conv2dLayer(layer = layer,
                act = act,
                shape = [filter_size, filter_size, in_channels, out_channels],
                strides=[1, stride, stride, 1],
                padding=padding,
                W_init = w_init,
                b_init = b_init,
                name =name)   

def conv2d_transpose(input, out_channels, filter_size, stride, act=None, padding='SAME', name='conv2d_transpose'):
    batch, height, width, in_channels = input.outputs.get_shape().as_list()
    output_shape = [batch, height*stride, width*stride, out_channels]
    return DeConv2dLayer(
        layer = input,
        act = act,
        shape = [filter_size, filter_size, out_channels, in_channels],
        output_shape = output_shape,
        strides = [1, stride, stride, 1],
        padding = padding,
        name =name,
    )

def conv3d2(input, out_channels, filter_size=3, stride=1, padding='SAME', name='conv3d'):
    in_channels = input.get_shape().as_list()[-1]
    filter_shape = [filter_size, filter_size, filter_size, in_channels, out_channels]
    strides = [1, stride, stride, stride, 1]
    
    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight_conv3d', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name='bias_conv3d', shape=[out_channels], initializer=tf.constant_initializer(value=0.0))
        return tf.nn.conv3d(input, weight, strides, padding)

def upscale(layer, scale=2, only_z=False, name='upscale'):
    return SubVoxelConv(layer, scale=scale, only_z=only_z, n_out_channel=None, act=tf.identity, name=name)
    
def concat(layer, concat_dim=-1, name='concat'):
    return ConcatLayer(layer, concat_dim=concat_dim, name=name)        

def batch_norm(layer, act=tf.identity, is_train=True, gamma_init=g_init, name='bn'):  
    return BatchNormLayer(layer, act=act, is_train=is_train, gamma_init=gamma_init, name=name)

def max_pool3d(x, filter_size=3, stride=2, padding='SAME', name='maxpool3d'):
    filter_size=[filter_size,filter_size,filter_size]
    strides=[stride,stride,stride]
    return MaxPool3d(x, filter_size=filter_size, strides=strides, padding=padding, name=name)
    
def prelu(x, name='prelu'):
    w_shape = x.get_shape()[-1]
    with tf.variable_scope(name):
        alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=tf.constant_initializer(value=0.0) )
        out = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
        return out



class LReluLayer(Layer):
    
    def __init__(self, layer=None, alpha=0.2, name='leaky_relu'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        
        with tf.variable_scope(name):
            self.outputs = tf.nn.leaky_relu(self.inputs, alpha=alpha)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )

class ReluLayer(Layer):
    
    def __init__(self, layer=None, name='relu'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        
        with tf.variable_scope(name):
            self.outputs = tf.nn.relu(self.inputs)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        
def _interpolateXY(data, ratio, nchannels_out=1):
   """
   interpolate height and width 
   
   Parameters
   data [batch_size, depth, height, width, nchannels]
   """
   assert len(data.shape) == 5, "incompatible data dimension"
   bsize, d, h, w, c = data.get_shape().as_list()
   assert c == (ratio**2) * nchannels_out, "nchannels_out: %d , nchannels_in: %d, ratio: %d\n" % (nchannels_out, c, ratio)
   
   Xs_ = tf.split(data, ratio, 4) # split channels
   Xr_ = tf.concat(Xs_, 3)        # concat at width dimension  
   #print(bsize, d, h*ratio, w*ratio, nchannels_out)
   X_ = tf.reshape(Xr_, (bsize, d, h*ratio, w*ratio, int(nchannels_out)))
   return X_
   
def SubVoxelConv(net, scale=2, only_z=False, n_out_channel=None, act=tf.identity, name='subpixel_conv3d'):    
    
    """
    Inflate pixles in [channels] into [depth height width] to interpola the 3-d image 
    Parameters
    net : [batch depth height width channels]
    
    """
    scope_name = tf.get_variable_scope().name
    if scope_name:
        name = scope_name + '/' + name

    

    def _PS(X, r, n_out_channel, z_only=False):
        if n_out_channel >= 1:
            batch_size, depth, height, width, channels = X.get_shape().as_list()
            batch_size = tf.shape(X)[0]
            Xs = tf.split(X, r, 4)
            
            if z_only is False:
                i = 0
                for subx in Xs:
                    Xs[i] = _interpolateXY(subx, scale, nchannels_out=int(Xs[i].get_shape()[-1]) / (scale**2))
                    i += 1
            
            Xr = tf.concat(Xs, 2)  # concat at height dimension
            if z_only is False:
                X = tf.reshape(Xr, (batch_size, depth*scale, height*scale, width*scale, n_out_channel))
            else:
                X = tf.reshape(Xr, (batch_size, depth*scale, height, width, n_out_channel))
            
        return X


    inputs = net.outputs
    
    
    down_factor = scale ** 3
    if only_z:
        down_factor = scale

    if n_out_channel is None:
        assert int(inputs.get_shape()[-1]) / down_factor % 1 == 0, "nchannels_in == ratio^3 * nchannels_out"
        n_out_channel = int(int(inputs.get_shape()[-1]) / down_factor)
        
    print("  SubvoxelConv  %s : scale: %d n_out_channel: %s act: %s " % (name, scale, n_out_channel, act.__name__))  
    
    net_new = Layer(inputs, name=name)
    with tf.variable_scope(name) as vs:
        net_new.outputs = act(_PS(inputs, r=scale, n_out_channel=n_out_channel, z_only=only_z))
    
    net_new.all_layers = list(net.all_layers)
    net_new.all_params = list(net.all_params)
    net_new.all_drop = dict(net.all_drop)
    net_new.all_layers.extend( [net_new.outputs] ) 
    return net_new



'''
def deep_feature_extractor(input, reuse=False, name="dfe"):
    n_layers_encoding = 3
    n_channels = 64
    n_channels_in = input.get_shape().as_list()[-1]
    features = []
    encoding = []
    
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(input)
        for i in range(1, n_layers_encoding + 1):
            n = conv3d(n, out_channels=n_channels*i, filter_size=3, stride=2, act=tf.nn.relu, padding='SAME', name='conv%d' % i)
            encoding.append(n)
            features.append(n.outputs)
        
        
        for i in range(1, n_layers_encoding + 1):
            c = n_channels*(n_layers_encoding - i)
            if c == 0:
                c = 32
            n = conv3d_transpose(n, out_channels=c, filter_size=3, stride=2, act=tf.nn.relu, padding='SAME', name='conv3d_transpose%d' % i )
            if n_layers_encoding - i > 0:
                n = concat([encoding[n_layers_encoding - i - 1], n], name='concat%d' % i)
        n = conv3d(n, out_channels=n_channels_in, filter_size=3, stride=1, act=tf.tanh, padding='SAME', name='out')
        
        return n, features

def deep_feature_loss(features_img, features_ref):
    assert len(features_img) == len(features_ref)
    diff = []   
    for i in range(0, len(features_img)):
        f_img = features_img[i] 
        f_ref = features_ref[i]
        f_diff = mean_squared_error(f_img, f_ref)
        diff.append(f_diff)
    diff_array = np.asarray(diff)
    diff_mean = diff_array.sum() / len(features_ref)
    return diff_mean
    
def bottleneck(layer, is_train=True, name='bottleneck'):
    nchannels = layer.outputs.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        n = conv3d(layer, shape=[1, 1, 1, nchannels, 16], name='reduce')
        n = batch_norm(n, is_train=is_train, name='reduce/bn')
        n = conv3d(n, shape=[3,3,3,16,16], name='mid')
        n = batch_norm(n, is_train=is_train, name='mid/bn')
        n = conv3d(n, shape=[1,1,1,16,nchannels], name='expand')
        return n
        
def res_block(layer, is_train=True, name='block'):
    with tf.variable_scope(name):
        #n = conv3d(layer, shape=[3,3,3,64,64], name='conv3d1')
        n = bottleneck(layer, is_train=is_train, name='bottleneck1')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='bn1')
        #n = conv3d(n, shape=[3,3,3,64,64], name='conv3d2')
        n = bottleneck(n, is_train=is_train, name='bottleneck2')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train,  name='bn2')
        n = ElementwiseLayer([layer, n], combine_fn=tf.add)
        return n

def res_blocks(n, n_blocks=8, out_channels=64, is_train=False, name='res_blocks'):
    with tf.variable_scope(name):
        temp = n
        # Residual Blocks
        for i in range(n_blocks):
            n = res_block(n, is_train=is_train, name='block%d' % i) 
            
        n = conv3d(n, shape=[3,3,3,64,out_channels], name='conv3d2')
        n = batch_norm(n, is_train=is_train, name='bn2')
        n = ElementwiseLayer([n, temp], combine_fn=tf.add)
        return n    
   

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data, z_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    
    z_data = np.expand_dims(z_data, axis=-1)
    z_data = np.expand_dims(z_data, axis=-1)
    
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    z = tf.constant(z_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2 + z**2)/(2.0*sigma**2)))

    window = g / tf.reduce_sum(g)
    return window


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, depth, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width, depth)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    mu1 = tf.nn.conv3d(padded_img1, window, strides=[1,1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv3d(padded_img2, window, strides=[1,1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2

    sigma1_sq = tf.nn.conv3d(paddedimg11, window, strides=[1,1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv3d(paddedimg22, window, strides=[1,1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv3d(paddedimg12, window, strides=[1,1,1,1,1], padding='VALID') - mu1_mu2
    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:
        cs_map_value = tf.clip_by_value((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim_resize(img1, img2, weights=None, return_ssim_map=None, filter_size=11, filter_sigma=1.5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    level = len(weights)
    assert return_ssim_map is None or return_ssim_map < level
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    _, h, w, _ = img1.get_shape().as_list()
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
        if return_ssim_map == l:
            return_ssim_map = tf.image.resize_images(ssim_map, size=(h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*(mssim[level-1]**weight[level-1])
    if return_ssim_map is not None:
        return value, return_ssim_map
    else:
        return value


def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1] # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] #[1, 1, 1, 1, 1] #
    level = len(weights)
    sigmas = [0.5]
    for i in range(level-1):
        sigmas.append(sigmas[-1]*2)
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma*4+1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)
    # list to tensor of dim D+1
    value = mssim[level-1]**weight[level-1]
    for l in range(level):
        value = value * (mcs[l]**weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value
'''