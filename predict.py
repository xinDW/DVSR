import numpy as np
from utils import normalize_percentile, _raise

class Predictor:
    def __init__(self, factor, model):
        self.factor = factor
        self.model  = model
        self.dtype  = None

    def __normalize(self, im, low=0.2, high=99.8):
        self.dtype = im.dtype
        return normalize_percentile(im, low=low, high=high)

    def __reverse_norm(self, im):
        # max_val = 65535. if self.dtype == np.uint16 else 255
        # eps = 1e-10
        # im  = (im - np.min(im)) / (np.max(im) - np.min(im) + eps) * max_val
        # return im.astype(self.dtype)

        max_val = 65535.
        eps = 1e-10
        im  = (im - np.min(im)) / (np.max(im) - np.min(im) + eps) * max_val
        return im.astype(np.uint16)


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
        
        overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_size)]

        overlap = [0 if b == s else i for i, b, s in zip(overlap, block_size, im.shape)] # no overlap along the dims where the image size equal to the block size

        block_size = [b - 2 * i for b, i in zip(block_size, overlap)]                    # real block size when inference

        overlap    = [int(i) for i in overlap]
        block_size = [int(i) for i in block_size]

        print('='*66)
        print('block size (overlap excluded) : {} overlap : {}\n'.format(block_size, overlap))

        return block_size, overlap

    def __region_iter(self, im, blk_size, overlap, factor):
        """
        Params:
            -im: ndarray in dims of [depth, height, width]
        """
        im_size = im.shape
        
        anchors = [(z, y, x) for z in range(overlap[0], im_size[0] - overlap[0], blk_size[0]) 
            for y in range(overlap[1], im_size[1] - overlap[1], blk_size[1])
            for x in range(overlap[2], im_size[2] - overlap[2], blk_size[2]) ]

        for i, anchor in enumerate(anchors):
            begin = [p - c for p, c in zip(anchor, overlap)]
            end   = [p + b + c for p, b, c in zip(anchor, blk_size, overlap)]
            yield [slice(b, e) for b, e in zip (begin, end)], \
            [slice((b + c // 2) * factor, (e - c // 2) * factor) for b, e, c, in zip(begin, end, overlap)], \
            [slice((c // 2)  * factor, (b + c + c // 2) * factor) for b, c in zip(blk_size, overlap)]
                            
    def predict(self, im, block_size, overlap, low=0.2, high=99.8):
        block_size, overlap = self.__check_dims(im, block_size, overlap)
        im = self.__normalize(im, low=low, high=high)
        factor  = self.factor

        sr_size = [s * factor for s in im.shape]
        sr      = np.zeros(sr_size)

        for src, dst, in_blk in self.__region_iter(im, block_size, overlap, factor):
            # print('source: {}  dst: {}  valid: {} '.format(src, dst, in_blk))
            begin = [i.start for i in src]
            end   = [i.stop for i in src]
            if not all(i <= j for i, j in zip(end, im.shape)):
                continue

            print('\r {}-{} in {}  ' .format(begin, end, im.shape), end='')
            block                  = im[tuple(src)]
            block                  = self.__predict_block(block)
            sr[tuple(dst)]         = block[tuple(in_blk)]
        print('')
        return self.__reverse_norm(sr)

