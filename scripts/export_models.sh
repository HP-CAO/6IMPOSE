#!/usr/bin/env sh

rm -r ./models/cps_models
mkdir ./models/cps_models

#python main.py --config ./logs/frontiers_stapler/config.json --id frontiers_stapler_export --params trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false --force --weights ./models/frontiers_stapler/best_model/model --mode export
#python main.py --config ./logs/frontiers_pliers/config.json --id frontiers_pliers_export --params trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false --force --weights ./models/frontiers_pliers/best_model/model --mode export
#python main.py --config ./logs/frontiers_wrench_13/config.json --id frontiers_wrench_export --params trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false --force --weights ./models/frontiers_wrench_13/best_model/model --mode export
#python main.py --config ./logs/frontiers_cpsglue/config.json --id frontiers_cpsglue_export --params trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false --force --weights ./models/frontiers_cpsglue/best_model/model --mode export
#python main.py --config ./logs/frontiers_chew_toy/config.json --id frontiers_schew_toy_export --params trainer_params/distribute_train_device 4 monitor_params/sim2real_eval false --force --weights ./models/frontiers_chew_toy/best_model/model --mode export
#
#mv ./models/frontiers_stapler_export/ ./models/cps_models/
#mv ./models/frontiers_pliers_export/ ./models/cps_models/
#mv ./models/frontiers_wrench_export/ ./models/cps_models/
#mv ./models/frontiers_cpsglue_export/ ./models/cps_models/
#mv ./models/frontiers_schew_toy_export/ ./models/cps_models/


python main.py --config ./logs/frontiers_stapler_512_sup/config.json --id frontiers_stapler_512_export --params trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false --force --weights ./models/frontiers_stapler_512_sup/best_model/model --mode export
python main.py --config ./logs/frontiers_pliers_512_sup/config.json --id frontiers_pliers_512_export --params trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false --force --weights ./models/frontiers_pliers_512_sup/best_model/model --mode export
#python main.py --config ./logs/frontiers_wrench_13_512/config.json --id frontiers_wrench_512_export --params trainer_params/distribute_train_device 1 monitor_params/sim2real_eval false --force --weights ./models/frontiers_wrench_13_512/25/model --mode export
python main.py --config ./logs/frontiers_cpsglue_512/config.json --id frontiers_cpsglue_512_export --params trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false --force --weights ./models/frontiers_cpsglue_512_sup/best_model/model --mode export
python main.py --config ./logs/frontiers_chew_toy_512/config.json --id frontiers_schew_toy_512_export --params trainer_params/distribute_train_device 7 monitor_params/sim2real_eval false --force --weights ./models/frontiers_chew_toy_512_sup/best_model/model --mode export

mv ./models/frontiers_stapler_512_export/ ./models/cps_models/
mv ./models/frontiers_pliers_512_export/ ./models/cps_models/
#mv ./models/frontiers_wrench_512_export/ ./models/cps_models/
mv ./models/frontiers_cpsglue_512_export/ ./models/cps_models/
mv ./models/frontiers_schew_toy_512_export/ ./models/cps_models/