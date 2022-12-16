#!/usr/bin/env sh

python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/cls_type cam darknet_params/cfg_file ./config/yolo_lm_config/yolov4-tiny-lm-cam.cfg trainer_params/distribute_train_device 4 &
python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/cls_type holepuncher darknet_params/cfg_file ./config/yolo_lm_config/yolov4-tiny-lm-holepuncher.cfg trainer_params/distribute_train_device 5 &
python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/cls_type lamp darknet_params/cfg_file ./config/yolo_lm_config/yolov4-tiny-lm-lamp.cfg trainer_params/distribute_train_device 6 &
python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type stapler darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-stapler.cfg trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false &

#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type cpsglue darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-cpsglue.cfg trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false
#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type wrench_13 darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-wrench_13.cfg trainer_params/distribute_train_device 5 monitor_params/sim2real_eval false
#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type chew_toy darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-chew_toy.cfg trainer_params/distribute_train_device 6 monitor_params/sim2real_eval false
#python main.py --config ./config/sim2real_darknet.json --id sim2real_darknet --params dataset_params/data_name cps dataset_params/cls_type pliers darknet_params/cfg_file ./config/yolo_cps_config/yolov4-tiny-cps-pliers.cfg trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false



