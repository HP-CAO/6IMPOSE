#!/usr/bin/env sh

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type ape
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type ape

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type benchvise
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type benchvise

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type cam
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type cam

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type can
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type can

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type cat
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type cat

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type driller
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type driller

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type duck
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type duck

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type eggbox
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type eggbox

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type glue
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type glue

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type holepuncher
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type holepuncher

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type iron
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type iron

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type lamp
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type lamp

python preprocessor.py --config config/sim2real_darknet.json --mode train --format darknet --params dataset_params/cls_type phone
python preprocessor.py --config config/sim2real_darknet.json --mode val --format darknet --params dataset_params/cls_type phone