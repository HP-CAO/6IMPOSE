#!/usr/bin/env sh

python main.py --config ./config/sim2real_pvn3d.json --id frontiers_cam --params dataset_params/cls_type cam trainer_params/distribute_train_device 0
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_holepuncher --params dataset_params/cls_type holepuncher trainer_params/distribute_train_device 1
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_lamp --params dataset_params/cls_type lamp trainer_params/distribute_train_device 2
python main.py --config ./config/sim2real_pvn3d.json --id frontiers_stapler --params dataset_params/data_name cps dataset_params/cls_type stapler trainer_params/distribute_train_device 3 monitor_params/sim2real_eval false

#
#python main.py --config ./config/sim2real_pvn3d.json --id frontiers_cpsglue --params dataset_params/data_name cps dataset_params/cls_type cpsglue trainer_params/distribute_train_device 0 monitor_params/sim2real_eval false
#python main.py --config ./config/sim2real_pvn3d.json --id frontiers_wrench_13 --params dataset_params/data_name cps dataset_params/cls_type wrench_13 trainer_params/distribute_train_device 1 monitor_params/sim2real_eval false
#python main.py --config ./config/sim2real_pvn3d.json --id frontiers_chew_toy --params dataset_params/data_name cps dataset_params/cls_type chew_toy trainer_params/distribute_train_device 2 monitor_params/sim2real_eval false
#python main.py --config ./config/sim2real_pvn3d.json --id frontiers_pliers --params dataset_params/data_name cps dataset_params/cls_type pliers trainer_params/distribute_train_device 3 monitor_params/sim2real_eval false
