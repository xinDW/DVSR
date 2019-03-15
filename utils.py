import imageio
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import SimpleITK as sitk

def read_all_images(path, z_range, format_out=True, factor=None):
    """
    Params:
        - format_out : see function 'transform' for details 
        - factor : useful when format_out == False
    return images in shape [n_images, depth, height, width, channels]
    """
    train_set = []
    img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
    
    if format_out == False:
        z_range = z_range // factor[0]
            
    for img_file in img_list:
        print(path + img_file)
        img = get_tiff_fn(img_file, path) 
        if format_out == False:
            assert factor != None
            img = transform(img, factor=factor, inverse=False) 

        if (img.dtype != np.float32):
            img = img.astype(np.float32, casting='unsafe')
            
        print(img.shape)

        depth = img.shape[0]
        for d in range(0, depth, z_range):
            if d + z_range <= depth:
                train_set.append(img[d:(d+z_range), ...])
    
    if (len(train_set) == 0):
        raise Exception("none of the images have been loaded, please check the config img_size and its real dimension")
    
    print('read %d from %s' % (len(train_set), path)) 
    train_set = np.asarray(train_set)
    print(train_set.shape)
    return train_set

## 
# TODO: is the parameter sess necessary ?
##
def load_and_assign_ckpt(sess, ckpt_file, net): 
     return tl.files.load_and_assign_npz(sess=sess, name=ckpt_file, network=net) 
    
def reformat(output):
    """
    Params : 
        output : [batch, depth, height, width, channels=1]
    """ 
    batch, depth, height, width, channels = output.get_shape().as_list();
    assert channels == 1
    center = width // 2
    sample = output[:,:,:,(center - 1) : (center + 2), :]
    resized_out = tf.image.resize_images(tf.squeeze(sample, [4]), size=[224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    resized_out = (resized_out + 1) / 2  # transform pixel value to [0, 1]
    
    return resized_out
    
def get_img_fn(file, path):    
    return imageio.imread(path + file)

def get_tiff_fn(file, path):
    """
    return volume in shape of [depth, height, width, channels=1]
    """
    image = imageio.volread(path + file) # [depth, height, width]
    image = image[..., np.newaxis]       # [depth, height, width, channels=1]
    image = normalize_fn(image)
    return image
    
def crop_img_fn(img, size, is_random=False):
    #img = crop(img, wrg=size[0], hrg=size[1], is_random=is_random)
    img = np.reshape(img, size)
    img = normalize_fn(img)
    return img

def normalize_fn(x):
    x = x / (255.0/2) - 1
    return x

def rearrange3d_fn(image):
    """
    re-arrange image of shape[depth, height, width, channels] into shape[height, width, depth]
    """
    
    image = np.squeeze(image); # remove channels dimension
    #print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth]) 
    for d in range(depth):
        image_re[:,:,d] = image[d,:,:]
    return image_re    

def _transform(image3d, factor, inverse=False):
    assert len(image3d.shape) == 4
    [d, h, w, c] = image3d.shape

    factors = 1
    for f in factor:
        factors *= f

    if inverse is False:
        assert d % factor[0] == 0 and h % factor[1] == 0 and w % factor[2] == 0

        transformed = np.zeros([d // factor[0], h // factor[1], w // factor[2], c * factors]) 

        for i in range(0, factor[0]):
            for j in range(0, factor[1]):
                for k in range(0, factor[2]):
                    idx = i * (factor[1]*factor[2]) + j * factor[2] + k
                    transformed[..., idx:idx+1] = image3d[i::factor[0], j::factor[1], k::factor[2], :]
    else:
        assert c == factors
        transformed = np.zeros([d*factor[0], h*factor[1], w*factor[2], 1])
        for i in range(0, factor[0]):
            for j in range(0, factor[1]):
                for k in range(0, factor[2]):
                    idx = i * (factor[1]*factor[2]) + j * factor[2] + k
                    transformed[i::factor[0], j::factor[1], k::factor[2], :] = image3d[..., idx:idx+1]

    return transformed


def transform(image3d, factor=[4,4,4], inverse=False):
    '''
    transform a 3D image with 1 channel into a multi-channel one, where the extra channels is filled by the spatial pixels
    Params:
        - image3d : 3D image with shape [depth, height, width, channels=1]
        - factor : super resolution factor for [depth, height, width] respectively
        - inverse : inverse the transform if true
    return:
        - transoformed image with shape[depth/factor[0], height/factor[1], width/factor[2], channels=factor**3]
    '''
    
    if len(image3d.shape) == 4: #[d, h, w, c]
        return _transform(image3d, factor, inverse=inverse)
    if len(image3d.shape) == 5: #[batch, d, h, w, c]
        
        for i in range(0, image3d.shape[0]):
            tmp = image3d[i]
            tmp = _transform(tmp, factor=factor, inverse=inverse)
            tmp = tmp[np.newaxis, ...]
            if i == 0:
                ret = tmp
            else:
                ret = np.concatenate((ret, tmp), axis=0)
    return ret

def _write3d(x, path, scale_pixel_value=True):
    """
    Params:
        -x : [depth, height, width, channels]
        -max_val : possible maximum pixel value (65535 for 16-bit or 255 for 8-bit)
    """
    if scale_pixel_value:
        x = x + 1.  #[0, 2]
        x = x * 65535. / 2.

    x = x.astype(np.uint16)
    stack = sitk.GetImageFromArray(x)
    #stack = sitk.Cast(stack, sitk.sitkUInt16)
    sitk.WriteImage(stack, path)
        
def write3d(x, path, scale_pixel_value=True):
    """
    Params:
        -x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
        -scale_pixel_value : scale pixels value to [0, 65535] is True
    """
    
    fragments = path.split('.')
    new_path = ''
    for i in range(len(fragments) - 1):
        new_path = new_path + fragments[i]
    
    #print(x.shape)
    dims = len(x.shape)
    
    if dims == 4:
        batch, height, width, n_channels = x.shape
        x_re = np.zeros([batch, n_channels, height, width, 1])
        for d in range(n_channels):
            slice = x[:,:,:,d]
            x_re[:,d,:,:,:] = slice[:,:,:,np.newaxis]
            
    elif dims == 5:
        x_re = x
    else:
        raise Exception('unsupported dims : %s' % str(x.shape))
    
    '''
    if bitdepth == 16:
        max_val = 65535.
    elif bitdepth == 8:
        max_val = 255.
    else :
        raise Exception('unsupported bitdepth : %d' % bitdepth)
    '''
    for index, image in enumerate(x_re):
        #print(image.shape)
        _write3d(image, new_path + '_' + str(index) + '.' + fragments[-1], scale_pixel_value)       
    