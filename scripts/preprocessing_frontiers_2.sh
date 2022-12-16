#!/usr/bin/env sh



python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type cpsglue
python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type cpsglue
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type cpsglue
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type cpsglue

#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type wrench_13
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type wrench_13
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type wrench_13
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type wrench_13
#
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type chew_toy
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type chew_toy
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type chew_toy
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type chew_toy
#
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type pliers
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type pliers
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type pliers
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type pliers
#
#
#
#

