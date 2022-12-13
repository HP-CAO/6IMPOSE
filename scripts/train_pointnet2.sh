#!/usr/bin/env sh

python main_pt2.py --config ./config/tf_pointnet2.json --id pt2_training_tf_seg_from_logits --force --params trainer_params/seg_from_logits true
python main_pt2.py --config ./config/tf_pointnet2.json --id pt2_training_tf --force
python main_pt2.py --config ./config/tf_pointnet2.json --id pt2_training_ori --params pt2_params/use_tf_interpolation false --force
