#!/usr/bin/env sh

python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode train --format darknet --params dataset_params/cls_type holepuncher
python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode val --format darknet --params dataset_params/cls_type holepuncher
python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode train --format tfrecord --params dataset_params/cls_type holepuncher
python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode val --format numpy --params dataset_params/cls_type holepuncher

python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode train --format darknet --params dataset_params/cls_type lamp
python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode val --format darknet --params dataset_params/cls_type lamp
python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode train --format tfrecord --params dataset_params/cls_type lamp
python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode val --format numpy --params dataset_params/cls_type lamp

python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type stapler
python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type stapler
python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type stapler
python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type stapler




