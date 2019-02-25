import tensorlayer as tl
import numpy as np
import os
from utils import get_tiff_fn

class Dataset:
    def __init__(self, train_hr_path, train_lr_path, train_mr_path, hr_size, lr_size):
        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        self.train_mr_path = train_mr_path

        self.lr_size = lr_size
        self.hr_size = hr_size
    
    def __load_training_data(self):
        def _read_images(path, img_size):
            
            """
            Params:
                
            return images in shape [n_images, depth, height, width, channels]
            """
            
            z_range = img_size[0] # the desired depth of the image
            images_set = []
            img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
            
            for img_file in img_list:
                img = get_tiff_fn(img_file, path) 
                if (img.dtype != np.float32):
                    img = img.astype(np.float32, casting='unsafe')
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

        self.training_data_lr = _read_images(self.train_lr_path, self.lr_size)
        self.training_data_mr = _read_images(self.train_mr_path, self.lr_size)
        self.training_data_hr = _read_images(self.train_hr_path, self.hr_size)
        
        assert self.training_data_hr.shape[0] == self.training_data_lr.shape[0]
        assert self.training_data_mr.shape[0] == self.training_data_lr.shape[0]
        return self.training_data_hr.shape[0]

    def prepare(self, batch_size, n_epochs):
        '''
        this function must be called after the Dataset instance is created
        '''
        if os.path.exists(self.train_lr_path) and os.path.exists(self.train_hr_path) and os.path.exists(self.train_mr_path):
            self.training_pair_num = self.__load_training_data()
        else:
            raise Exception('image data path doesn\'t exist')
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = batch_size
        self.epoch = 0

        print('HR dataset : %s\nLR dataset: %s\nMR dataset: %s\n' % (str(self.training_data_hr.shape), str(self.training_data_lr.shape), str(self.training_data_mr.shape)))
        return self.training_pair_num

    def for_eval(self):
        return self.training_data_hr[0 : self.batch_size], self.training_data_lr[0 : self.batch_size], self.training_data_mr[0 : self.batch_size]

    def hasNext(self):
        return True if self.epoch < self.n_epochs else False
             
    def iter(self):
       
        
        if self.epoch < self.n_epochs - 1:
            if self.cursor + self.batch_size >= self.training_pair_num:
                self.epoch += 1
                self.cursor = self.batch_size

            idx = self.cursor
            bth = self.batch_size

            self.cursor += bth

            return self.training_data_hr[idx : idx + bth], self.training_data_lr[idx : idx + bth], self.training_data_mr[idx : idx + bth], idx, self.epoch
                
        else:
            return None, None, None, self.cursor, self.epoch
