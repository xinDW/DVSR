import imageio
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import math
import scipy.io


__all__ = ['read_all_images',
    'load_and_assign_ckpt',
    'reformat',
    'get_tiff_fn',
    'rearrange3d_fn',
    'write3d',
    'interpolate3d',
    ]

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
        output : tf.Tensor, [batch, depth, height, width, channels=1]
    """ 
    _, _, _, width, channels = output.get_shape().as_list()
    assert channels == 1
    center = width // 2
    sample = output[:,:,:,(center - 1) : (center + 2), :]
    resized_out = tf.image.resize_images(tf.squeeze(sample, [4]), size=[224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    resized_out = (resized_out + 1) / 2  # transform pixel value to [0, 1]
    
    return resized_out

def get_tiff_fn(file, path):
    """
    return volume in shape of [depth, height, width, channels=1]
    """
    image = imageio.volread(path + file) # [depth, height, width]
    max_val = 255.
    if image.dtype == np.uint8:
        pass
    elif image.dtype == np.uint16:
        max_val = 65535.
    else:
        raise Exception('\nunsupported image bitdepth %s\n' % str(image.dtype))
    
    image = normalize_fn(image, max_val)
    image = image[..., np.newaxis]       # [depth, height, width, channels=1]
    return image

def generate_mr_fn(image, mode='ds', **kwargs) :
    """
    Params:
        -image: the source HR image, [depth, height, width, channels=1]
        -mode: 'ds'   -- down-sample only
               'blur' -- blur only
    """ 
    assert mode in ['ds', 'blur']
    depth, height, width, channels = image.shape
    if mode == 'ds':
        factor = kwargs['factor']
        tmp = np.zeros([depth//factor, height//factor, width//factor, channels])
        for i in range(0, depth, factor):
            d = i // factor
            tmp[d, :, :, :] = tmp[d, :, :, :] + image[i, ::factor, ::factor, :]
        tmp = tmp / factor
    else :
        tmp = image
    return tmp

def normalize_fn(x, max_val=255.):
    #max_val = 255. if bitdepth == 8 else 65535.
    x = x / (max_val/2) - 1
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

def _write3d(x, filename, scale_pixel_value=True):
    """
    Params:
        -x : [depth, height, width]
        -max_val : possible maximum pixel value (65535 for 16-bit or 255 for 8-bit)
    """
    if scale_pixel_value:
        x = x - np.min(x)
        #x = x + 1.  #[0, 2]
        x = x * 65535. / 2.

    x = x.astype(np.uint16)
    imageio.volwrite(filename, x)
    #stack = sitk.GetImageFromArray(x)
    #sitk.WriteImage(stack, filename)
        
def write3d(x, path, scale_pixel_value=True, savemat=False):
    """
    Params:
        -x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
        -scale_pixel_value : scale pixels value to [0, 65535] is True
        -savemat : if to save x as an extra .mat file.
    """
    
    
    
    #print(x.shape)
    new_path = ''
    fragments = path.split('.')
    for i in range(len(fragments) - 1):
        new_path = new_path + fragments[i]

    if savemat:
        save_mat(x, new_path)

    dims = len(x.shape)
    batch = x.shape[0]
    n_channels = x.shape[-1]
    
    if dims == 4:
        x_re = np.transpose(x, axes=[0, 1, 2, 3])
        for b in range(batch):
            x_re_b = x_re[b, ...]
            _write3d(x_re_b, new_path + '_{}.{}'.format(b, fragments[-1]) , scale_pixel_value)  
            
    elif dims == 5:
        x_re = x
        for b in range(batch):
            x_re_b = x_re[b, ...]
            for c in range(n_channels):
                x_re_c = x_re_b[..., c]
                _write3d(x_re_c, new_path + '_b{}_c{}.{}'.format(b, c, fragments[-1]) , scale_pixel_value)      
                #print(image.shape)
    else:
        raise Exception('unsupported dims : %s' % str(x.shape))

def save_mat(im, filename):
    """save the image as .mat file.
    """
    scipy.io.savemat(filename, mdict={'data' : im})

def interpolate3d(img, factor=4, order=1):
    """
    Params:
        -img : [batch, depth, height, width, channels] or [depth, height, width, channels]
    """
    from scipy.ndimage.interpolation import zoom
    if len(img.shape) == 5:
        zoom_factor = [1,factor,factor,factor,1]
    elif len(img.shape) == 4:
        zoom_factor = [factor,factor,factor,1]
    else:
        raise Exception("interpolate3d : unsupported image dims : %d" % len(img.shape))

    return zoom(img, zoom=zoom_factor, order=order)


'''
    def makeGaussianKernel1D(sigma):
    """
    params:
        -sigma : standard deviation of the 1-D Gaussian function, measured in `pixels`
    """
    kernel_radius = (int) (2 * sigma + 1)
    kernel = np.zeros(kernel_radius * 2 - 1)
    k_sum = 0.
    for i in range(0, kernel_radius * 2 - 1):
        x = i - kernel_radius
        kernel[i] = math.exp(-0.5 * x * x / sigma / sigma)
        k_sum += kernel[i]
    kernel = kernel / k_sum # normalization
    return kernel

def conv1d(line, kernel):
    """1-D convolution
    Params:
        -line : 1-D array, a row or colume from a image
        -kernel: 1-D normalized kernel
    return:
        convolved line
    """

    k_radius = kernel.size // 2
    conv = np.zeros(line.size)
    for i in range(0, line.size):
        tmp = 0.
        for j in range(-k_radius, k_radius + 1):
            idx = i + j
            if (idx >= 0 and idx < line.size):
                tmp += line[idx] * kernel[j + k_radius]
        conv[i] = tmp
    return conv

def gaussianBlur2D(image, kernel):
    """
    Perform a 2-D Gaussian blur by x-conv and y-conv seperately. The image channels remain unchanged.
    Params:
        -image: [height, width, channels=1]
        -kernel: 1-D Gaussian kernel
    """
    height, width, channels = image.shape
    blurred = np.zeros([height, width, channels])
    for c in range(0, channels):
        image_c = image[:,:, c]
        image_bc = np.zeros([height, width])
        for h in range(0, height):
            image_bc[h, :] = conv1d(image_c[h, :], kernel)
        for w in range(0, width):
            image_bc[:, w] = conv1d(image_bc[:, w], kernel)
    
        blurred[:,:,c] = image_bc
    
    return blurred

def gaussianBlur3D(image, sigma_xy, sigma_z):
    """ 
    Params:
        -image: [depth, height, width, channels] 
        -sigma: standard deviations of the 3D Gussian function, measured in pixels
    return: the blurred image
    """

    depth, height, width, channels = image.shape
    blurred = np.zeros([depth, height, width, channels])
    kernel_xy = makeGaussianKernel1D(sigma_xy)
    kernel_z = makeGaussianKernel1D(sigma_z)
    ## blur xy
    for d in range(0, depth):
        blurred[d, ...] = gaussianBlur2D(image[d, ...], kernel_xy)    
    ## blur z 
    for h in range(0, height):
        for w in range(0, width):
            for c in range(0, channels):
                blurred[:, h, w, c] = conv1d(blurred[:, h, w, c], kernel_z)
    return blurred
'''