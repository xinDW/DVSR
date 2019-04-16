import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
from utils import get_tiff_fn

class Dataset:
    def __init__(self, 
        lr_size, 
        hr_size,
        mr_size,
        train_lr_path,
        train_hr_path,
        train_mr_path,
        test_lr_path,
        test_hr_path,
        test_mr_path,
        valid_lr_path=None,
        dtype=np.float32):  
        
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.mr_size = mr_size

        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        self.train_mr_path = train_mr_path
        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
        self.test_mr_path = test_mr_path

        self.dtype = dtype

        ## if LR measurement is designated for validation during the trianing
        if valid_lr_path is not None:
            self.valid_lr_path = valid_lr_path
            self.hasValidation = True
        else:
            self.hasValidation = False

        self.prepared = False

    def _load_training_data(self):
        def _read_images(path, img_size, dtype=np.float32):
            
            """
            Params:
                
            return images in shape [n_images, depth, height, width, channels]
            """
            
            z_range = img_size[0] # the desired depth of the image
            images_set = []
            img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
            
            for img_file in img_list:
                img = get_tiff_fn(img_file, path) 
                if (img.dtype != dtype):
                    img = img.astype(dtype, casting='unsafe')
                print('%s : %s' % ((path + img_file), str(img.shape)))  

                depth = img.shape[0] # the actual depth of the image
                for d in range(0, depth, z_range):
                    if d + z_range <= depth:
                        images_set.append(img[d:(d+z_range), ...])
            
            if (len(images_set) == 0):
                raise Exception("none of the images have been loaded, please check the config img_size and its real dimension")
            
            print('read %d from %s' % (len(images_set), path)) 
            images_set = np.asarray(images_set)
        
            return images_set

        self.training_data_lr = _read_images(self.train_lr_path, self.lr_size, self.dtype)
        self.training_data_mr = _read_images(self.train_mr_path, self.mr_size)
        self.training_data_hr = _read_images(self.train_hr_path, self.hr_size)

        self.test_data_lr = _read_images(self.test_lr_path, self.lr_size, self.dtype)
        self.test_data_hr = _read_images(self.test_hr_path, self.hr_size)
        self.test_data_mr = _read_images(self.test_mr_path, self.mr_size)
        
        if self.hasValidation:
            self.valid_data_lr = _read_images(self.valid_lr_path, self.lr_size, self.dtype)
           # self.plchdr_lr_valid = tf.placeholder(self.dtype, shape=self.valid_data_lr.shape, name='valid_lr')

        assert self.training_data_hr.shape[0] == self.training_data_lr.shape[0]
        assert self.training_data_mr.shape[0] == self.training_data_lr.shape[0]

        return self.training_data_hr.shape[0]

    def prepare(self, batch_size, n_epochs):
        '''
        this function must be called after the Dataset instance is created
        '''
        if self.prepared == True:
            return self.training_pair_num

        if not os.path.exists(self.train_lr_path):
            raise Exception('lr training data path doesn\'t exist : %s' % self.train_lr_path)
        if not os.path.exists(self.train_hr_path):
            raise Exception('hr training data path doesn\'t exist : %s' % self.train_hr_path)
        if not os.path.exists(self.train_mr_path):
            raise Exception('mr training data path doesn\'t exist : %s' % self.train_mr_path)
        if self.hasValidation:
            if not os.path.exists(self.valid_lr_path):
                raise Exception('test data path doesn\'t exist : %s' % self.valid_lr_path)
        self.training_pair_num = self._load_training_data()
        
            
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = batch_size
        self.epoch = 1
        self.prepared = True
        
        print('HR dataset : %s\nLR dataset: %s\nMR dataset: %s\n' % (str(self.training_data_hr.shape), str(self.training_data_lr.shape), str(self.training_data_mr.shape)))

        return self.training_pair_num

    ## in case that the term "test" and "valid" are confusing:
    #  -test : test data follows the same probability distribution as the training dataset, thus a part of training data is used as the test data.
    #  -valid : validation data (also called development set) is used to choose a model, thus LR measurement is the valid date.
        
    def for_test(self):
        #return self.training_data_hr[0 : self.batch_size], self.training_data_lr[0 : self.batch_size], self.training_data_mr[0 : self.batch_size]
        return self.test_data_hr, self.test_data_lr, self.test_data_mr

    def for_valid(self):
        if self.hasValidation:
            return self.valid_data_lr
        else:
            raise Exception ('validation set not designated')

    def hasNext(self):
        return True if self.epoch < self.n_epochs else False
             
    def iter(self):
       
        
        if self.epoch < self.n_epochs:
            if self.cursor + self.batch_size > self.training_pair_num:
                self.epoch += 1
                self.cursor = 0

            idx = self.cursor
            bth = self.batch_size

            self.cursor += bth

            return self.training_data_hr[idx : idx + bth], self.training_data_lr[idx : idx + bth], self.training_data_mr[idx : idx + bth], idx, self.epoch
                
        else:
            return None, None, None, self.cursor, self.epoch

    def test_pair_nums(self):
        return self.test_data_lr.shape[0]

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