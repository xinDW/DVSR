import os

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import imageio

from utils import load_im, interpolate3d, _raise

class Dataset:
    def __init__(self, 
        lr_size, 
        hr_size, 
        train_lr_path,
        train_hr_path,
        test_lr_path=None,  # if None, the first 4 image pairs in the training set will be used as the test data
        test_hr_path=None,
        mr_size=None,
        train_mr_path=None,
        test_mr_path=None,
        valid_lr_path=None,
        dtype=np.float32,
        normalization = 'fixed',
        keep_all_blocks = False,
        transforms = None, # [trans_fn_for_lr, trans_fn_for_hr] or None
        shuffle = True,
        **kwargs           # keyword arguments for transform function
        ):  
        
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.mr_size = mr_size

        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        self.train_mr_path = train_mr_path
        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
        self.test_mr_path = test_mr_path
        self.valid_lr_path = valid_lr_path

        normalization in ['fixed', 'percentile'] or _raise(ValueError('unknown normalization mode: %' % normalization))
        self.normalize = normalization

        self.keep_all_blocks = keep_all_blocks
        self.transforms = transforms if transforms is not None else [None, None]
        self.transforms_args = kwargs
        self.dtype = dtype
        self.shuffle = shuffle
        
        ## if LR measurement is designated for validation during the trianing
        self.hasValidation = False
        if valid_lr_path is not None:
            self.hasValidation = True

        self.hasMR = False
        if train_mr_path is not None:
            self.hasMR = True
        
        self.hasTest = False
        if test_hr_path is not None:
            self.hasTest = True

        self.prepared = False

    def _check_inputs(self):
        print("checking training data dims ... ")
        hr_im_list = sorted(tl.files.load_file_list(path=self.train_hr_path, regx='.*.tif', printable=False))
        lr_im_list = sorted(tl.files.load_file_list(path=self.train_lr_path, regx='.*.tif', printable=False))
        len(hr_im_list) == len(lr_im_list) or _raise(ValueError("Num of HR and LR not equal"))

        for hr_file, lr_file in zip(hr_im_list, lr_im_list):
            hr = imageio.volread(os.path.join(self.train_hr_path, hr_file))
            lr = imageio.volread(os.path.join(self.train_lr_path, lr_file))
            # print('checking dims: \n%s %s\n%s %s' % (hr_file, str(hr.shape), lr_file, str(lr.shape)))
            if 'factor' not in dir(self):
                self.factor = hr.shape[0] // lr.shape[0]
            valid_dim = [self.factor == hs / ls for hs, ls in zip(hr.shape, lr.shape)]
            if not all(valid_dim):
                raise(ValueError('dims mismatch: \n%s %s\n%s %s' % (hr_file, str(hr.shape), lr_file, str(lr.shape))) )

    def _load_training_data(self, shuffle=True):

        def _shuffle_in_unison(arr1, arr2):
            """shuffle elements in arr1 and arr2 in unison along the leading dimension 
            Params:
                -arr1, arr2: np.ndarray
                    must be in the same size in the leading dimension
            """
            assert (len(arr1) == len(arr2))
            new_idx = np.random.permutation(len(arr1)) 
            return arr1[new_idx], arr2[new_idx]

        def _shuffle_index(len):
            new_index = np.random.permutation(len)
            return new_index

        def _get_im_blocks(path, block_size, dtype=np.float32, transform=None, keep_all=True, keep_list=None, **kwargs):
            
            """laod image volume and crop into small blocks for training dataset.
            Params:
                -block_size : [depth height width channels]  
                -transform : transformation function applied to the loaded image
                -keep_all: boolean, whether to kepp all the blocks
                -keep_list : numpy list, index of block to be kept, useful when keep_all is False
                -kwargs : key-word args for transform fn

            return images in shape [n_images, depth, height, width, channels]
            """
            
            depth, height, width, _ = block_size # the desired image block size
            blocks = []
            im_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
            # im_list = sorted(tl.files.load_file_list(path=path, regx='.*.mat', printable=False))

            block_idx = -1
            idx_saved = []
            keep_list_cursor = 0

            for im_file in im_list:
                im = load_im(path + im_file, normalize=self.normalize) 
                
                if (im.dtype != dtype):
                    im = im.astype(dtype, casting='unsafe')
                print('\r%s : %s ' % ((path + im_file), str(im.shape)), end='')  
                
                if transform is not None:
                    im = transform(im, **kwargs)
                    print('transfrom: %s' % str(im.shape), end = '')
                d_real, h_real, w_real, _ = im.shape # the actual size of the image
                max_val = np.percentile(im, 98)

                
                for d in range(0, d_real, depth):
                    for h in range(0, h_real, height):
                        for w in range(0, w_real, width):
                            if d + depth > d_real or h + height > h_real or w + width > w_real :
                                # out of image bounds
                                continue
                            
                            block = im[d:(d+depth), h:(h+height), w:(w+width), :]
                            block_idx += 1

                            if not keep_all:
                                if keep_list is None:
                                    if (np.max(block) > max_val * 0.7):
                                        blocks.append(block)
                                        idx_saved.append(block_idx)
                                else:
                                    if (keep_list_cursor < len(keep_list) and block_idx == keep_list[keep_list_cursor]):
                                        blocks.append(block)
                                        keep_list_cursor += 1
                            else:
                                blocks.append(block)
                            
            print('\nload %d of size %s from %s' % (len(blocks), str(block_size), path)) 
            blocks = np.asarray(blocks)

            keep_list = idx_saved if keep_list is None else keep_list
            return blocks, keep_list



        self._check_inputs()
        self.training_data_hr, valid_indices = _get_im_blocks(self.train_hr_path, 
                                                            self.hr_size, 
                                                            transform=self.transforms[1], keep_all=self.keep_all_blocks)
        len(self.training_data_hr) != 0 or _raise(Exception("none of the HRs have been loaded, please check the image size ({} desired)".format(str(self.hr_size))) )


        #self.training_data_lr = _get_im_blocks(self.train_lr_path, self.hr_size, self.dtype, transform=interpolate3d)
        self.training_data_lr, _ = _get_im_blocks(self.train_lr_path, 
                                                    self.lr_size, 
                                                    keep_all=self.keep_all_blocks, 
                                                    keep_list=valid_indices, 
                                                    transform=self.transforms[0], **self.transforms_args)
        len(self.training_data_lr) != 0 or _raise(Exception("none of the LRs have been loaded, please check the image size ({} desired)".format(str(self.lr_size))) )
        self.training_data_hr.shape[0] == self.training_data_lr.shape[0] or _raise(ValueError("num of LR blocks and HR blocks not equal"))
        
        

        self.test_data_split = int(len(self.training_data_hr) * 0.2)
        if self.hasTest:
            self.test_data_lr, _ = _get_im_blocks(self.test_lr_path, self.lr_size, transform=self.transforms[0], **self.transforms_args)
            self.test_data_hr, _ = _get_im_blocks(self.test_hr_path, self.hr_size, transform=self.transforms[1])
            self.test_data_split = 0 

        if self.hasMR:
            self.training_data_mr, _ = _get_im_blocks(self.train_mr_path, self.mr_size, keep_all=self.keep_all_blocks, keep_list=valid_indices)
            self.training_data_mr.shape[0] == self.training_data_lr.shape[0] or _raise(ValueError("num of MR blocks and LR blocks not equal"))
            if self.hasTest:
                self.test_data_mr, _ = _get_im_blocks(self.test_mr_path, self.mr_size)
        
        if self.hasValidation:
            self.valid_data_lr, _ = _get_im_blocks(self.valid_lr_path, self.lr_size, transform=self.transforms[0], **self.transforms_args)
           # self.plchdr_lr_valid = tf.placeholder(self.dtype, shape=self.valid_data_lr.shape, name='valid_lr')
        
        return self.training_data_hr.shape[0]

    def prepare(self, batch_size, n_epochs):
        '''
        this function must be called after the Dataset instance is created
        '''
        if self.prepared == True:
            return self.training_pair_num

        os.path.exists(self.train_lr_path) or _raise (Exception('lr training data path doesn\'t exist : %s' % self.train_lr_path))
        os.path.exists(self.train_hr_path) or _raise (Exception('hr training data path doesn\'t exist : %s' % self.train_hr_path))
        if (self.hasMR) and (not os.path.exists(self.train_mr_path)):
            raise Exception('mr training data path doesn\'t exist : %s' % self.train_mr_path)
        if self.hasValidation and (not os.path.exists(self.valid_lr_path)):
            raise Exception('validation data path doesn\'t exist : %s' % self.valid_lr_path)

        self.training_pair_num = self._load_training_data()

        # if self.test_data_split >= self.training_pair_num:
        #     self.test_data_split = self.training_pair_num // 2
            
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = self.test_data_split
        self.epoch = 0
        self.prepared = True
        
        print('HR dataset: %s\nLR dataset: %s' % (str(self.training_data_hr.shape), str(self.training_data_lr.shape)))
        if self.hasMR:
            print('MR dataset: %s' % str(self.training_data_mr.shape))
        print()
        return self.training_pair_num - self.test_data_split

    ## in case that the term "test" and "valid" are confusing:
    #  -test : test data follows the same probability distribution as the training dataset, thus a part of training data is used as the test data.
    #  -valid : validation data (also called development set) is used to choose a model, thus LR measurement is the valid date.

    def __check_prepared(self):
        self.prepared or _raise (Exception('Dataset.prepare() must be called') )

    def for_test(self):
        #return self.training_data_hr[0 : self.batch_size], self.training_data_lr[0 : self.batch_size], self.training_data_mr[0 : self.batch_size]
        self.__check_prepared()
        
        if self.hasTest:
            if self.hasMR:
                return self.test_data_hr, self.test_data_lr, self.test_data_mr
            else:
                return self.test_data_hr, self.test_data_lr, None
        else:
            n = self.test_data_split
            if self.hasMR:
                return self.training_data_hr[0 : n], self.training_data_lr[0 : n], self.training_data_mr[0 : n]
            else:
                return self.training_data_hr[0 : n], self.training_data_lr[0 : n], None

    def test_data_iter(self, epoch):
        if self.hasTest:
            if self.hasMR:
                return self.test_data_hr, self.test_data_lr, self.test_data_mr
            else:
                return self.test_data_hr, self.test_data_lr, None
        else:
            n = self.test_data_split
            if self.hasMR:
                return self.training_data_hr[0 : n], self.training_data_lr[0 : n], self.training_data_mr[0 : n]
            else:
                return self.training_data_hr[0 : n], self.training_data_lr[0 : n], None

    def for_valid(self):
        self.__check_prepared()

        if self.hasValidation:
            return self.valid_data_lr
        else:
            raise Exception ('validation set not designated')

    def hasNext(self):
        return True if self.epoch <= self.n_epochs else False
             
    def iter(self):
       
        n_t = self.test_data_split
        if self.epoch <= self.n_epochs:
            if self.cursor + self.batch_size > self.training_pair_num:
                self.epoch += 1
                self.cursor = n_t

            idx = self.cursor
            bth = self.batch_size

            self.cursor += bth
            idx_disp = idx - n_t # begin with 0
            if self.hasMR:
                return self.training_data_hr[idx : idx + bth], self.training_data_lr[idx : idx + bth], self.training_data_mr[idx : idx + bth], idx_disp, self.epoch
            else :
                return self.training_data_hr[idx : idx + bth], self.training_data_lr[idx : idx + bth], None, idx_disp, self.epoch

        raise Exception('epoch idx out of bounds : %d / %d' %(self.epoch, self.n_epochs))

    def test_pair_nums(self):
        return (self.test_data_lr.shape[0] if self.hasTest else self.test_data_split)

    def reset(self, n_epochs):
        self.n_epochs = n_epochs
        self.cursor = self.test_data_split
        self.epoch = 0

    '''
    def for_training(self, sess):
        if self.use_tf_data_api is False:
            raise Exception("for_training() can only be called when use_tf_data_api is True")

        self.plchdr_hr_train = tf.placeholder(self.dtype, shape=self.training_data_hr.shape, name='train_hr')
        self.plchdr_lr_train = tf.placeholder(self.dtype, shape=self.training_data_lr.shape, name='train_lr')
        self.plchdr_mr_train = tf.placeholder(self.dtype, shape=self.training_data_mr.shape, name='train_mr')
        self.plchdr_hr_test = tf.placeholder(self.dtype, shape=self.training_data_hr.shape, name='train_hr')
        self.plchdr_lr_test = tf.placeholder(self.dtype, shape=self.training_data_lr.shape, name='train_lr')
        self.plchdr_mr_test = tf.placeholder(self.dtype, shape=self.training_data_mr.shape, name='train_mr')  

        dataset = tf.data.Dataset().from_tensor_slices((self.plchdr_hr_train, self.plchdr_lr_train, self.plchdr_mr_train))
        #dataset = dataset.batch(self.batch_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next = iterator.get_next()
        sess.run(iterator.make_initializer(dataset), feed_dict={self.plchdr_hr_train : self.training_data_hr, self.plchdr_lr_train : self.training_data_lr, self.plchdr_mr_train : self.training_data_mr})
        return next
    '''