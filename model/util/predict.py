import os
import time

import numpy as np
import tensorflow as tf

from utils import normalize_percentile, normalize_max, _raise, get_file_list, imread2d, imwrite2d

class Model:
    def __init__(self, net, sess, input_plchdr):
        self.net    = net if isinstance(net, tf.Tensor) else net.outputs
        self.sess   = sess
        self.plchdr = input_plchdr
    def predict(self, input):
        len(input.shape) == 3 or _raise(ValueError(''))
        input = input[np.newaxis, ..., np.newaxis]
        feed_dict = {self.plchdr : input}
        out       = self.sess.run(self.net, feed_dict)
        return np.squeeze(out, axis=(0, -1)) 

    def recycle(self):
        # if self.sess is not None:
        #     self.sess.close()
        self.sess = None

class Predictor:
    def __init__(self, factor, model, half_precision=False):
        self.factor       = factor
        self.model        = model
        self.model_dtype  = np.float16 if half_precision else np.float32

    def __normalize_percentile(self, im, low=10, high=99.8):
        return normalize_percentile(im.astype(self.model_dtype), low=low, high=high)

    def __normalize_fixed(self, im, max_v=None):
        if max_v is None:
            max_v = self.__get_bitdepth_max(im)
        else:
            max_v = max_v * self.__get_bitdepth_max(im)
            
        im = im.astype(self.model_dtype)
        return normalize_max(im, max_v)

    def __get_bitdepth_max(self, im, dtype=None):
        max_v = 0
        im_dtype = im.dtype if dtype is None else dtype

        if im_dtype == np.uint8:
            max_v = 255.
        elif im_dtype == np.uint16:
            max_v = 65535.
        else :
            max_v = np.max(im)
        return max_v

    def __reverse_norm(self, im, dtype=np.uint16, normalize_mode='fixed'):
        
        bitdepth_max =  self.__get_bitdepth_max(im, dtype)
        
        eps = 1e-6
        max_ = np.max(im)
        # min_ = np.min(im)
        min_ = np.percentile(im, 0.2)
        im   = np.clip(im, min_, max_)
        im = (im - min_) / (max_ - min_ + eps) * bitdepth_max
    
        print('reverse normalization to [%.4f, %.4f]' % (np.min(im), np.max(im)))
        return im.astype(dtype)


    def __predict_block(self, block):  
        return self.model.predict(block)

    def __check_dims(self, im, block_size, overlap):
        # To-Do: deal with float block_size and overlap
        def __broadcast(val):
            val = [val for i in im.shape] if np.isscalar(val) else val
            return val

        len(im.shape) == 3 or _raise(ValueError('the input image must be in shape [depth, height, width]'))
        block_size = __broadcast(block_size)
        overlap    = __broadcast(overlap)

        len(block_size) == len(im.shape) or _raise(ValueError("ndim of block_size ({}) mismatch that of image size ({})".format(block_size, im.shape)))
        len(overlap) == len(im.shape) or _raise(ValueError("ndim of overlap ({}) mismatch that of image size ({})".format(overlap, im.shape)))
        
        # block_size = [b if b <= i else i for b, i in zip(block_size, im.shape)]

        overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_size)]
        overlap = [0 if b >= s else i for i, b, s in zip(overlap, block_size, im.shape)] # no overlap along the dims where the image size equal to the block size
        overlap = [i if i % 2 == 0 else i + 1 for i in overlap]                          # overlap must be even number

        block_size = [b - 2 * i for b, i in zip(block_size, overlap)]                    # real block size when inference
        
        overlap    = [int(i) for i in overlap]
        block_size = [int(i) for i in block_size]


        print('block size (overlap excluded) : {} overlap : {}'.format(block_size, overlap))

        return block_size, overlap

    def _padding_block(self, im, blk_size, overlap):
        grid_dim       = [int(np.ceil(float(i) / b)) for i, o, b in zip(im.shape, overlap, blk_size)]
        im_size_padded = [(g * b + b if o != 0 else g * b) for g, b, o in zip(grid_dim, blk_size, overlap)]
       
        im_wrapped     = np.ones(im_size_padded, dtype=self.model_dtype) * np.min(im) 

        valid_region    = [slice(o // 2, o // 2 + i) for o, i in zip(overlap, im.shape)]
        sr_valid_region = [slice(o // 2 * self.factor, (o // 2 + i) * self.factor) for o, i in zip(overlap, im.shape)]
        print('raw image size : {}, wrapped into : {}'.format(im.shape, im_size_padded))
        print('valid region index: {} ({} after SR)'.format(valid_region, sr_valid_region))
        
        im_wrapped[tuple(valid_region)] = im
        
        return im_wrapped, sr_valid_region

    def __region_iter(self, im, blk_size, overlap, factor):
        """
        Params:
            -im: ndarray in dims of [depth, height, width]
        """
        im_size = im.shape
        
        anchors = [(z, y, x) for z in range(overlap[0], im_size[0], blk_size[0]) 
            for y in range(overlap[1], im_size[1], blk_size[1])
            for x in range(overlap[2], im_size[2], blk_size[2]) ]

        for i, anchor in enumerate(anchors):
            # revised_overlap = [0 if a == i else i for a, i in zip(anchor, overlap)]
            begin = [p - c for p, c in zip(anchor, overlap)]
            end   = [p + b + c for p, b, c in zip(anchor, blk_size, overlap)]
            yield [slice(b, e) for b, e in zip (begin, end)], \
            [slice((b + c // 2) * factor, (e - c // 2) * factor) for b, e, c, in zip(begin, end, overlap)], \
            [slice((c // 2)  * factor, (b + c + c // 2) * factor) for b, c in zip(blk_size, overlap)]

    def predict_without_norm(self, im, block_size, overlap):  
        block_size, overlap = self.__check_dims(im, block_size, overlap)
        factor  = self.factor

        
        im_wrapped, valid_region_idx = self._padding_block(im, block_size, overlap)
        sr_size = [s * factor for s in im_wrapped.shape]
        sr      = np.zeros(sr_size, dtype=self.model_dtype)

        for src, dst, in_blk in self.__region_iter(im_wrapped, block_size, overlap, factor):
            # print('source: {}  dst: {}  valid: {} '.format(src, dst, in_blk))
            begin = [i.start for i in src]
            end   = [i.stop for i in src]

            if not all(i <= j for i, j in zip(end, im_wrapped.shape)):
                continue

            print('\revaluating {}-{} in {}  ' .format(begin, end, im_wrapped.shape), end='')
            block                  = im_wrapped[tuple(src)]
            block                  = self.__predict_block(block)
            sr[tuple(dst)]         = block[tuple(in_blk)]

        print('')
        return sr[tuple(valid_region_idx)]

    def predict(self, im, block_size, overlap, normalization='fixed', **kwargs):
        normalization in ['fixed', 'percentile'] or _raise(ValueError('unknown normailze mode:%s' % normalization))
        norm_fn = self.__normalize_fixed if normalization == 'fixed' else self.__normalize_percentile

        im_dtype = im.dtype
        im_dtype in [np.uint8, np.uint16] or _raise(ValueError('unknown image dtype:%s' % im_dtype))
        im = norm_fn(im, **kwargs)
        
    
        print('normalized to [%.4f, %.4f]' % (np.min(im), np.max(im)))
        sr = self.predict_without_norm(im, block_size, overlap)
        return self.__reverse_norm(sr, normalize_mode=normalization)


class LargeDataPredictor:
    def __init__(self, data_path, saving_path, factor, model, block_size, overlap, half_precision):
        self.data_path   = data_path
        self.saving_path = saving_path
        self.block_size  = block_size
        self.overlap     = overlap
        self.predictor   = Predictor(factor=factor, model=model, half_precision=half_precision)
        self.model_dtype = np.float16 if half_precision else np.float32

    def _prepare(self):
        
        file_list = get_file_list(self.data_path, regx='.tif')
        self.im_file_list = [os.path.join(self.data_path, im_file) for im_file in file_list]
        interval          = 50

        self._sample_and_get_statistics(self.im_file_list, interval, low=0.2, high=95)

        
    def predict(self):
        self._prepare()
        
        block_depth = self.block_size[0]
        block       = np.zeros([block_depth, self.height, self.width], dtype=self.model_dtype)
       
        margin_z    = 5
        slice_iter  = margin_z

        start_time = time.time()
        for slice_iter in range(margin_z, self.n_slices - block_depth, block_depth - 2 * margin_z):
            
            block[0 : 2 * margin_z, ...] = block[block_depth - 2 * margin_z : block_depth, ...]
            for i in range(2 * margin_z, block_depth):
                print('\rloading slice %d' % (slice_iter + i), end='')
                im = imread2d(self.im_file_list[slice_iter + i])
                block[i, ...] = self._normalize(im, self.p_low, self.p_high)
            
            print()
            sr_block = self.predictor.predict_without_norm(block, block_size=self.block_size, overlap=self.overlap)
            for i in range(margin_z, block_depth - margin_z):
                idx = i - margin_z + slice_iter 
                im  = self._reverse_normlize(sr_block[i,...])
                imwrite2d(im, os.path.join(self.saving_path, '%05d.tif' % idx))

            time_elapsed = (time.time() - start_time) / 60
            time_left    = (self.n_slices - slice_iter) / slice_iter * time_elapsed
            print('%d slices processed in %.2f mins; %.2f extra mins required' % (slice_iter, time_elapsed, time_left))

    def _sample_and_get_statistics(self, file_list, interval=100, low=2, high=99.8):
        n_slices = len(file_list)
        n_slices > interval or _raise(ValueError('n_slices %d < sample interval %d' % (n_slices, interval)))

        samples = [imread2d(file_list[i]) for i in range(0, n_slices, interval)]
        # samples = imread2d(file_list[0]) 
        samples = np.asarray(samples, dtype=self.model_dtype)

        print('samples volume : %s' % str(samples.shape))
        _, h, w = samples.shape
        # h, w = samples.shape

        p_low  = np.percentile(samples, low)
        p_high = np.percentile(samples, high)
        # p_low  = 0
        # p_high = 8000
        print('normalization thres: %.2f, %.2f' % (p_low, p_high))

        self.n_slices = n_slices
        self.p_low    = p_low
        self.p_high   = p_high
        self.width    = w
        self.height   = h
        

    def _normalize(self, block, p_low, p_high):
        # return (block.astype(self.model_dtype) - p_low) / (p_high - p_low)
        return (block.astype(self.model_dtype) / (p_high / 2) - 1)

    def _reverse_normlize(self, block, min_v=0, max_v=255):  
        # print('global:[%.6f,%.6f], local [%.6f, %.6f]   ' % (min_v, max_v, np.min(block), np.max(block)), end='')
        # block = ((block - min_v) / (max_v - min_v) * 255).astype(np.uint8)
        # print('after rescale : [%.6f, %.6f]   ' % (np.min(block), np.max(block)))
        # return block
        block = np.tanh(block)
        block = (block + 1) * 255 / 2.
        block = block.astype(np.uint8)
        return block