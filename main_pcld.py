import argparse
from utils import *


def main(p, mode):

    network = MainPcld(p)

    if mode == "train":
        network.train()
    else:
        assert p.monitor_params.weights_path is not None, 'no pre-trained model provided'
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='config/pvn3d_test.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    # './model/model_name/model_name'
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='Choose the mode, train or test')

    args = parser.parse_args()

    if args.config is None:
        exit("config file needed")

    params = read_config(args.config)

    import tensorflow as tf

    if not params.trainer_params.distribute_training:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[-1], True)
        except:
            exit("GPU allocated failed")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    from lib.main_pcldnet import MainPcld, MainPcldParams

    if args.generate_config:

        network_params = MainPcldParams()

        generate_config(network_params, "config/default_{}.json".format("pcld_net"))
        exit("config file generated")

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
