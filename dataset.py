import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
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

        def _read_images(path, img_size, dtype=np.float32, transform=None, **kwargs):
            
            """
            Params:
                -img_size : [depth height width channels]  
                -transform : transformation function applied to the loaded image
                -kwargs : key-word args for transform fn

            return images in shape [n_images, depth, height, width, channels]
            """
            
            depth, height, width, _ = img_size # the desired image size
            images_set = []
            #img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
            img_list = sorted(tl.files.load_file_list(path=path, regx='.*.mat', printable=False))
            for img_file in img_list:
                img = load_im(path + img_file) 
                
                if (img.dtype != dtype):
                    img = img.astype(dtype, casting='unsafe')
                print('%s : %s' % ((path + img_file), str(img.shape)))  
                
                if transform is not None:
                    img = transform(img, **kwargs)
                    print(img.shape)
                d_real, h_real, w_real, _ = img.shape # the actual size of the image
                for d in range(0, d_real, depth):
                    if d + depth <= d_real:
                        for h in range(0, h_real, height):
                            for w in range(0, w_real, width):
                                if h + height <= h_real and w + width <= w_real :
                                    images_set.append(img[d:(d+depth), h:(h+height), w:(w+width), :])
        
            
            print('read %d from %s' % (len(images_set), path)) 
            images_set = np.asarray(images_set)
        
            return images_set

        #self.training_data_lr = _read_images(self.train_lr_path, self.hr_size, self.dtype, transform=interpolate3d)
        self.training_data_lr = _read_images(self.train_lr_path, self.lr_size, transform=self.transforms[0], **self.transforms_args)
        len(self.training_data_lr) != 0 or _raise(Exception("none of the LRs have been loaded, please check the image size ({} desired)".format(str(self.lr_size))) )

        self.training_data_hr = _read_images(self.train_hr_path, self.hr_size, transform=self.transforms[1])
        len(self.training_data_hr) != 0 or _raise(Exception("none of the HRs have been loaded, please check the image size ({} desired)".format(str(self.hr_size))) )

        self.test_data_split = int(len(self.training_data_hr) * 0.2)
        if self.hasTest:
            self.test_data_lr = _read_images(self.test_lr_path, self.lr_size, transform=self.transforms[0], **self.transforms_args)
            self.test_data_hr = _read_images(self.test_hr_path, self.hr_size, transform=self.transforms[1])
            self.test_data_split = 0 

        if self.hasMR:
            self.training_data_mr = _read_images(self.train_mr_path, self.mr_size)
            if self.hasTest:
                self.test_data_mr = _read_images(self.test_mr_path, self.mr_size)
        
        if self.hasValidation:
            self.valid_data_lr = _read_images(self.valid_lr_path, self.lr_size, transform=self.transforms[0], **self.transforms_args)
           # self.plchdr_lr_valid = tf.placeholder(self.dtype, shape=self.valid_data_lr.shape, name='valid_lr')

        assert self.training_data_hr.shape[0] == self.training_data_lr.shape[0]
        if self.hasMR:
            assert self.training_data_mr.shape[0] == self.training_data_lr.shape[0]

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
        
        print('HR dataset: %s\nLR dataset: %s\n' % (str(self.training_data_hr.shape), str(self.training_data_lr.shape)))
        if self.hasMR:
            print('MR dataset: %s\n' % str(self.training_data_mr.shape))
        
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