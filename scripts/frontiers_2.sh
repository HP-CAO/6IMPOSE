#!/usr/bin/env sh

# todo before this we need to generate key-points for new objects

#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type cpsglue
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type cpsglue
#python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type cpsglue
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type cpsglue
#
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type wrench_13
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type wrench_13
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type wrench_13
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type wrench_13
#
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type chew_toy
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type chew_toy
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type chew_toy
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type chew_toy
#
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type pliers
#python preprocessor.py --config config/sim2real_darknet_proc_2.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type pliers
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type pliers
python preprocessor.py --config config/sim2real_pvn3d_proc_2.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type pliers



#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type cpsglue darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-cpsglue.cfg trainer_params/distribute_train_device 0 monitor_params/sim2real_eval false --force &
#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type wrench_13 darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-wrench_13.cfg trainer_params/distribute_train_device 1 monitor_params/sim2real_eval false --force &
#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type chew_toy darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-chew_toy.cfg trainer_params/distribute_train_device 2 monitor_params/sim2real_eval false --force &
#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type pliers darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-pliers.cfg trainer_params/distribute_train_device 3 monitor_params/sim2real_eval false --force &

python main.py --config ./config/sim2real_pvn3d.json --id frontiers_cpsglue --params dataset_params/data_name cps dataset_params/cls_type cpsglue trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false --force &
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_wrench_13 --params dataset_params/data_name cps dataset_params/cls_type wrench_13 trainer_params/distribute_train_device 5 monitor_params/sim2real_eval false --force &
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_chew_toy --params dataset_params/data_name cps dataset_params/cls_type chew_toy trainer_params/distribute_train_device 6 monitor_params/sim2real_eval false --force &
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_pliers --params dataset_params/data_name cps dataset_params/cls_type pliers trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false --force
