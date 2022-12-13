import argparse
import os
from utils import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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

    args = parser.parse_args()

    mode = args.mode

    if args.generate_config:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # don't need gpu for generating config
        from lib.factory import NetworkFactory

        network_factory = NetworkFactory()
        network_params = network_factory.get_default_params()
        generate_config(network_params, "config/default_{}.json".format(args.net))
        exit("config file generated")

    if args.config is None:
        exit("config file needed")

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.monitor_params.model_name = args.id
        params.monitor_params.log_file_name = args.id

    if args.force:
        params.monitor_params.force_override = True

    params.monitor_params.weights_path = args.weights

    params.monitor_params.mode = args.mode

    if params.trainer_params.distribute_training:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(x) for x in params.trainer_params.distribute_train_device])
    # else:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.trainer_params.distribute_train_device[0])

    import tensorflow as tf

    for dev in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(dev, True)

    if mode != 'train':
        params.monitor_params.write_log = False

    from lib.factory import NetworkFactory, DatasetFactory

    network_factory = NetworkFactory(params)
    ds_factory = DatasetFactory(params)  # make accessible for commandline usage
    network = network_factory.get_network()
    if mode == "train":
        network.train()
    elif mode == 'export':
        network.initial_trainer_and_model()
        network.export_model()
    elif mode == 'val':
        network.initial_trainer_and_model()
        network.performance_evaluation(0)
    elif mode == 'test':
        """ for commandline testing """
        network.initial_trainer_and_model()
    elif mode == 'demo':
        network.run_demo()
