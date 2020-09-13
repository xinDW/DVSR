
# DSP-Net 

DSP-Net for 3-D super resolution of microscopy images.  

## Requirements

DSP-Net is built with Python and Tensorflow. Technically there are no limits to the operation system to run the code, but Windows system is recommonded, on which the software has been tested.

The inference process of the DSP-Net can run using the CPU only, but could be inefficiently. A powerful CUDA-enabled GPU device that can speed up the inference is highly recommended. 

The inference process has been tested with:

 * Windows 10 pro (version 1903)
 * Python 3.6.7 (64 bit)
 * tensorflow 1.15.0
 * Intel Core i7-5930K CPU @3.50GHz
 * Nvidia GeForce RTX 2080 Ti


The inference of the example data `example_data/brain/LR/cerebellum.tif` took 12s and 350s with and without GPU promotion in the tested platform.


## Install

1. Install python 3.6 
2. (Optional) If your computer has a CUDA-enabled GPU, install the CUDA and CUDNN of the proper version.
3. Download the DSP_Demo.zip and unpack it. The directory tree should be: 

```  
DSP-Demo   
    .
    ├── config.py
    ├── dataset.py
    ├── eval.py
    ├── model
    ├── requirements.txt
    ├── utils.py
    ├── example_data
        └── brain
        └── cell
```

4. Install the dependencies using pip:

```
pip install -r requirements.txt
```

5. (Optional) If you have had the CUDA environment and the CUDNN library installed properly, run:

```
pip install tensorflow-gpu=1.15.0
```

The installation takes about 5 minutes in the tested platform. The time could be longer due to the network states.

## Usage

### Train

We provide a group of training data of micro-tubulins for the users to hand on quickly. Run:

```
python train.py
```

To train the DSP-Net on your own data:
1. Prepare your dataset that contains HRs, LRs, and MRs in three seperate folders. We highly recommend that the corresponding HR, LR and MR having the same file name.
2. Specify the paths to the training data: 
    ```
    # ./config.py

    train_lr_img_path = "path-to-your-lr-data/"
    train_hr_img_path = "path-to-your-hr-data/"
    train_mr_img_path = "path-to-your-mr-data/"
    ```
3. Name your training with a distinguishable **label** in `config.py`, which contains at least the following information:
    * The net architechture to be used;
    * The resolution enhancement factor (must be 1, 2 or 4);s
    * The normalization method to pre-process the input images;
    * The loss function.

    for example, a label named 
    ```
    # ./config.py

    label = 'tubulin_2stage-dpbn+rdn_factor-2_norm-percentile_loss-mse'
    ```
    means that :
    * The training dataset consists of images of tubulin structures;
    * The network to be used is a 2-stage one, with the dbpn as the 1st stage and the rdn as the second stage;
    * The resolution enhancement factor is 2;
    * A percentile normalization would be applied to the input data for preprocessing;
    * The loss function is MSE.

    Note that these **options** consists of a left-hand side (e.g., "factor"), a right-hand side (e.g., "2") and a dash "-" between them. Options are divided with a underdash "_". The available options are as follow:

    | lfs | rhs | meaning |
    |-----|-----|---------|
    |2stage | dpbn+rdn| dual-stage net consists of a dbpn and a rdn|
    |       | denoise+rdn| dual-stage net consists of a denoising subnet and a rdn|
    |1stage | dbpn| 1-stage net using dpbn architecture|
    |       | rdn | 1-stage net using rdn architecture|
    |       | unet| 1-stage net using u-net architecture from [CARE](https://github.com/CSBDeep/CSBDeep)|
    |factor | 1   | The output is the same size as the input, for u-net only|
    |       | 2   | The output is twice the size in each dimension as the input. |
    |       | 4   | The output is 4 times the size in each dimension as the input. |
    |norm   | percentile| use percentile normalization for data preprocessing. Helpful when the input signal is very weak|
    |       | fixed     | normalize the data x by (x / (MAX / 2) - 1), where MAX is the possible maximum of the bitdepth(e.g, 255 for 8-bit image, 65536 for 16-bit image) |
    |loss   | mse | use mean-square-error as the loss function|
    |       | mae | use the mean-absolute-error as the loss function|


4. Run the training by
    ```
    python train.py
    ```

    The training ends when the epoch reaches its maximum value (500 by dafalut), or can be early-stopped manually by the user. 

### Inference

This toturial contains example data of mouse brains and cells (see example_data/):
```
.
├── example_data
    └── brain
        └── LR
            └── cerebellum_3.2x_bessel.tif                (3.2x bessel-sheet cerebellum data)
            └── vessel_3.2x_bessel_cross-sample.tif       (3.2x bessel-sheet brain vessel data, for cross-sample application )
            └── cortex_3.2x_confocal_cross-mode.tif       (3.2x confocal brain data of the cortex region , for cross-mode application )
        └── expected_outputs
    └── cell
        └── LR
            └── microtube_60x_bessel.tif                  (60x bessel-beam images of the microtube of the U2OS cells)
            └── ER_60x_bessel_cross-sample.tif            (60x bessel-beam images of the endoplasmic reticulum of the cell, for cross-sample application)
        └── expected_outputs

```

The expected outputs by the DSP-Net of each input LR can be found in the corresponding 'expected_outputs' directory (due to the size limit, only MIPs of expected outputs are provided. 

To run the DSP-inference, use :

```
python eval.py 
```
The 3-D tiff images under the directory specified by `valid_lr_img_path` in `your_config.py` will be loaded and processed, using the model corresponding to the `label` in `your_config.py`.

Currently only 3-D tiff file in 8-bit and 16-bit are supported. The outputs will be saved in a subfolder named by the label under the input image directory.

## Troubleshoot 
:point_down:

|Problem| Solution |
|-------|----------|
|There are not any signals in the outputs.| Try to use `tf.identity` instead of `tf.nn.tanh` as the activation functions of the final layer.|
|Out of memory when training| Use small batch size (`config.TRAIN.batch_size`) and block size (`train_img_size_lr`) in `config.py`.|
|Out of memory at inference| Use small block size (`valid_lr_img_size`) in config.py. |
|Grid patterns in the outputs| The input image is processed block by block and stitched together by the overlap. Try to use a bigger block size (`valid_block_size`) or overlap (`valid_block_overlap`). Make sure that block_size * overlap is even.|