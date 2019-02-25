import tensorflow as tf
from tensorlayer.layers import Layer

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
        
def interpolateXY(data, ratio, nchannels_out=1):
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
   
def SubpixelConv3d(net, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_conv3d'):    
    
    """
    Inflate pixles in [channels] into [depth height width] to interpola the 3-d image 
    Parameters
    net : [batch depth height width channels]
    
    """
    scope_name = tf.get_variable_scope().name
    if scope_name:
        name = scope_name + '/' + name
        
    def _PS(X, r, n_out_channel):
        if n_out_channel >= 1:
            assert int(X.get_shape()[-1]) == (r ** 3) * n_out_channel, "n_channels_in == scale^3 * n_channels_out"
            batch_size, depth, height, width, channels = X.get_shape().as_list()
            batch_size = tf.shape(X)[0]
            Xs = tf.split(X, r, 4)
            
            i = 0;
            for subx in Xs:
                Xs[i] = interpolateXY(subx, scale, nchannels_out=int(Xs[i].get_shape()[-1]) / (scale**2))
                i += 1
            
            Xr = tf.concat(Xs, 2)  # concat at height dimension
            print(batch_size)
            X = tf.reshape(Xr, (batch_size, depth*scale, height*scale, width*scale, n_out_channel))
            
        return X
    
    inputs = net.outputs
    
    if n_out_channel is None:
        assert int(inputs.get_shape()[-1]) / (scale ** 3) % 1 == 0, "nchannels_in == ratio^3 * nchannels_out"
        n_out_channel = int(int(inputs.get_shape()[-1]) / (scale ** 3))
        
    print("  SubpixelConv3d  %s : scale: %d n_out_channel: %s act: %s " % (name, scale, n_out_channel, act.__name__))  
    
    net_new = Layer(inputs, name=name)
    with tf.variable_scope(name) as vs:
        net_new.outputs = act(_PS(inputs, r=scale, n_out_channel=n_out_channel))
    
    net_new.all_layers = list(net.all_layers)
    net_new.all_params = list(net.all_params)
    net_new.all_drop = dict(net.all_drop)
    net_new.all_layers.extend( [net_new.outputs] ) 
    return net_new