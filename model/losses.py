import tensorflow as tf
import numpy as np

def sobel_edges(input):
    '''
    find the edges of the input image, using the bulit-in tf function

    Params: 
        -input : tensor of shape [batch, depth, height, width, channels]
    return:
        -tensor of the edges: [batch, height, width, depth]
    '''
    # transpose the image shape into [batch, h, w, d] to meet the requirement of tf.image.sobel_edges
    img = tf.squeeze(tf.transpose(input, perm=[0,2,3,1,4]), axis=-1) 
    
    # the last dim holds the dx and dy results respectively
    edges_xy = tf.image.sobel_edges(img)
    #edges = tf.sqrt(tf.reduce_sum(tf.square(edges_xy), axis=-1))
    
    return edges_xy
    
def sobel_edges2(input):
    '''
    custom sobel operator for edges detection
    Params 
        - input : tensor of shape [batch, depth, height, width, channels=1]
    '''
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter_x = tf.constant(filter_x, dtype=tf.float32, name='sobel_x')
    filter_y = tf.constant(filter_y, dtype=tf.float32, name='sobel_y')
    
    filter_x = tf.reshape(filter_x, [3,3,1,1])
    filter_y = tf.reshape(filter_y, [3,3,1,1])
    
    batch, depth, height, width, _ = input.shape.as_list()
    
    with tf.name_scope('sobel'):
        
        for d in range(0, depth):
            edges_x = tf.nn.conv2d(input[:,d,:,:,:], filter_x, strides=(1,1,1,1), padding='SAME', name='edge_x')
            edges_y = tf.nn.conv2d(input[:,d,:,:,:], filter_y, strides=(1,1,1,1), padding='SAME', name='edge_y')
            edges = tf.sqrt(tf.square(edges_x) + tf.square(edges_y))
            
            edges = tf.expand_dims(edges, axis=1) 
            if d == 0:
                stack = edges
            else : 
                stack = tf.concat([stack, edges], axis=1)
            '''
            edges_x_t = tf.nn.conv2d(input[:,d,:,:,:], filter_x, strides=(1,1,1,1), padding='SAME', name='edge_x')
            edges_y_t = tf.nn.conv2d(input[:,d,:,:,:], filter_y, strides=(1,1,1,1), padding='SAME', name='edge_y')
            
            edges_x = tf.expand_dims(edges_x_t, axis=1) 
            edges_y = tf.expand_dims(edges_y_t, axis=1) 
            if d == 0:
                stack_x = edges_x
                stack_y = edges_y
            else : 
                stack_x = tf.concat([stack_x, edges_x], axis=1)
                stack_y = tf.concat([stack_y, edges_y], axis=1)
            stack = tf.sqrt(tf.square(stack_x) + tf.square(stack_y))
            '''
        return  stack 
        
        
def l2_loss(image, reference):
  with tf.variable_scope('l2_loss'):
    return tf.reduce_mean(tf.squared_difference(image, reference))
    
def l1_loss(image, reference):
  with tf.variable_scope('l1_loss'):
    return tf.reduce_mean(tf.abs(image - reference))
    
def edges_loss(image, reference):
    '''
    params: 
        -image : tensor of shape [batch, depth, height, width, channels], the output of DVSR
        -reference : same shape as the image
    '''
    with tf.variable_scope('edges_loss'):
        edges_sr = sobel_edges(image)
        edges_hr = sobel_edges(reference)
        
        #return tf.reduce_mean(tf.abs(edges_sr - edges_hr))
        return l2_loss(edges_sr, edges_hr)
        
def img_gradient_loss(image, reference):
    '''
    params: 
        -image : tensor of shape [batch, depth, height, width, channels]
        -reference : same shape as the image
    '''
    with tf.variable_scope('gradient_loss'):
        img = tf.squeeze(tf.transpose(image, perm=[0,2,3,1,4]), axis=-1) 
        ref = tf.squeeze(tf.transpose(reference, perm=[0,2,3,1,4]), axis=-1) 
        grad_i = tf.image.image_gradients(img)
        grad_r = tf.image.image_gradients(ref)
        g_loss = tf.reduce_mean(tf.squared_difference(grad_i, grad_r))
        return g_loss 
           
def mean_squared_error(target, output, is_mean=False, name="mean_squared_error"):
    """ Return the TensorFlow expression of mean-square-error (L2) of two batch of data.

    Parameters
    ----------
    output : 2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, w, h] or [batch_size, w, h, c].
    target : 2D, 3D or 4D tensor.
    is_mean : boolean, if True, use ``tf.reduce_mean`` to compute the loss of one data, otherwise, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    """
    output = tf.cast(output, tf.float32)
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:   # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3: # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4: # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
                
        elif output.get_shape().ndims == 5: # [batch_size, depth, height, width, channels]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3, 4]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3, 4]))
        else:
            raise Exception("Unknow dimension")
        return mse

def cross_entropy(labels, probs):
  return tf.reduce_mean(tf.sigmoid(labels) *  tf.log_sigmoid(probs) + tf.sigmoid(1 - labels) * tf.log_sigmoid(1 - probs))
        