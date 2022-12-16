##!/usr/bin/env sh
#
#python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode train --format darknet --params dataset_params/cls_type holepuncher
#python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode val --format darknet --params dataset_params/cls_type holepuncher
#python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode train --format tfrecord --params dataset_params/cls_type holepuncher
#python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode val --format numpy --params dataset_params/cls_type holepuncher
#
#python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode train --format darknet --params dataset_params/cls_type lamp
#python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode val --format darknet --params dataset_params/cls_type lamp
#python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode train --format tfrecord --params dataset_params/cls_type lamp
#python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode val --format numpy --params dataset_params/cls_type lamp
#
#python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode train --format darknet --params dataset_params/data_name cps dataset_params/cls_type stapler
#python preprocessor.py --config config/sim2real_darknet_proc_1.json --mode val --format darknet --params dataset_params/data_name cps dataset_params/cls_type stapler
#python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode train --format tfrecord --params dataset_params/data_name cps dataset_params/cls_type stapler
#python preprocessor.py --config config/sim2real_pvn3d_proc_1.json --mode val --format numpy --params dataset_params/data_name cps dataset_params/cls_type stapler


python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/cls_type cam darknet_params/cfg_file ./config/yolo_lm_config/yolov4-tiny-lm-cam.cfg trainer_params/distribute_train_device 4  --force &
python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/cls_type holepuncher darknet_params/cfg_file ./config/yolo_lm_config/yolov4-tiny-lm-holepuncher.cfg trainer_params/distribute_train_device 5 --force  &
python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/cls_type lamp darknet_params/cfg_file ./config/yolo_lm_config/yolov4-tiny-lm-lamp.cfg trainer_params/distribute_train_device 6 --force  &
python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type stapler darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-stapler.cfg trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false --force &

python main.py --config ./config/sim2real_pvn3d.json --id frontiers_cam --params dataset_params/cls_type cam trainer_params/distribute_train_device 0 --force &
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_holepuncher --params dataset_params/cls_type holepuncher trainer_params/distribute_train_device 1 --force &
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_lamp --params dataset_params/cls_type lamp trainer_params/distribute_train_device 2 --force &
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_stapler --params dataset_params/data_name cps dataset_params/cls_type stapler trainer_params/distribute_train_device 3 monitor_params/sim2real_eval false --force
