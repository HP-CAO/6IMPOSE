import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utils import *


def main(p, mode):
    if mode == "train":
        from lib.main_pvn import MainPvn3d
        network = MainPvn3d(p)
        network.train()
    else:
        assert p.monitor_params.weights_path is not None, 'no pre-trained model provided'
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='config/default_pvn3d.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    # './model/model_name/model_name'
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='Choose the mode, train or test')
    parser.add_argument('--net', default='pvn3d', help='network type')

    args = parser.parse_args()

    if args.generate_config:
        from lib.main_pvn import MainPvn3dParams

        network_params = MainPvn3dParams()

        generate_config(network_params, "config/default_{}.json".format(args.net))
        exit("config file generated")
    else:
        params = read_config(args.config)

        if args.config is None:
            exit("config file needed")

        if args.params is not None:
            params = override_params(params, args.params)

        if args.id is not None:
            params.monitor_params.model_name = args.id
            params.monitor_params.log_file_name = args.id

        if args.force:
            params.monitor_params.force_override = True

        params.monitor_params.weights_path = args.weights

        params.monitor_params.mode = args.mode

        main(params, args.mode)
