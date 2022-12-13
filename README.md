# Sim2real 6D pose estimation using YOLO and PVN3D
## Overview
In this work, we develop a real-time two-stage 6D pose estimation approach by integrating the object detector YOLO-V4-tiny [[1]](#1) and the 6D pose estimation algorithm PVN3D [[2]](#2) for time sensitive robotics applications.

![Alt text](diagram.jpg?raw=true "")

*A two-stage pose estimation approach shows the object detection with YOLO-tiny to localize the object of interest at the first stage, followed by the 6D object pose estimation with PVN3D-tiny at the second stage.*
## Setting up
### Basics: 
This project is using the following settings:

- Ubuntu: 20.04
- CUDA: 11.0
- Tensorflow: 2.6.2
- python: >3.6.5 (Recommend using conda python>=3.6.2)

### Compile Pointnet++ layers
Download [Pointnet++ tensorflow 2.0 layers](https://github.com/dgriffiths3/pointnet2-tensorflow2)
and save it under /pvn3d/lib/. Then compile the pointnet++ layers following the introduction. 
Some modifications needed:
- Before compiling, ensure the CUDA_ROOT path in ```./tf_ops/compile_ops.sh``` is correct
- After compiling, you maybe need to modify the path in ```./pnet2_layers/cpp_modules.py```

### Datasets
- *blender*: synthetically generated Dataset using Blender.
- *linemod*: Kinect V1 RGBD dataset with object poses. We download the [LM dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) as in original [pvn3d implementation](https://github.com/ethnhe/PVN3D)

### Networks
- *pvn3d*: Pointwise keypoint voting network for pose estimation from RGBD. The Original PyTorch implementation can be found in [pvn3d repo](https://github.com/ethnhe/PVN3D)
- *yolov4*: YOLO-V4 is taken from the [Darknet implementation](https://github.com/AlexeyAB/darknet)

## Inspection
### Usage
This tool allows the inspection and validation of the datasets. Specify which dataset and what to inspect to generate an output.json file containing the relevant statistics and preview randomly sampled datapoints.
```
inspect_datasets.py [-h] [--mode MODE] [--num_imgs NUM_IMGS]
                           datasets [datasets ...]

positional arguments:
  datasets             format as <data_set>/<data_name>/<cls_type>. 
                       Add */aug* to perform augmentation on that dataset

optional arguments:
  -h, --help           show this help message and exit
  --mode MODE          [rgb|bbox|mask|depth|normals|statistics] define what to
                       inspect. join multiple ops with /. Default is all
  --num_imgs NUM_IMGS  Number of images to inspect
```

## Preprocessing
### Usage
Preprocess datasets to speed up training and evaluation. Set `use_preprocessed=True` in the config to utilize the preprocessed data. Preprocess the validation split in the `numpy` format and the training split in the `tfrecord` format. Use `darknet` format to generate a training dataset for YOLO (darknet implementation). Set further settings in the respective config file. To overwrite config parameters from the commandline, use e.g. `--params dataset_params/clstype duck`
```
preprocessor.py [-h] [--config CONFIG] [--num_workers NUM_WORKERS]
                       [--params [PARAMS [PARAMS ...]]] [--mode MODE]
                       [--format FORMAT]
arguments:
  --mode MODE          [train|val|full] decide image split of dataset
  --format FORMAT      [tfrecord|numpy|darknet] decide save format (WIP for darknet ajust path
                        in script)
```

## Training
### Usage
```
main.py [-h] [--generate_config] [--config CONFIG] [--id ID] [--force]
               [--weights WEIGHTS] [--params [PARAMS [PARAMS ...]]]
               [--mode MODE]
arguments:
  --mode:              [train] for now only training is supported
```

## General Notes
- Either use `CUDA_VISIBLE_DEVICES=<gpus>` from the commandline or make sure the correct GPUS are set in the main scripts
- Use NetworkFactory and DatasetFactory from pvn3d.factory to instanciate MainNetworks and 'Dataset's, tf.data.Datasets from generator or tfrecord according to params from config file -> refer to main.py file for usage

- Use to experiment in command line 
```
CUDA_VISIBLE_DEVICES=-1 python -i main.py --config <your_config> --mode test --params monitor_params/write_log false
```

## References
<a id="1">[1]</a> 
Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).

<a id="1">[2]</a> 
He, Yisheng, et al. "Pvn3d: A deep point-wise 3d keypoints voting network for 6dof pose estimation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
