import argparse
from utils import read_config, write_config, override_params
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='./config', help='Path to config files')
    parser.add_argument('--params', nargs='*', default=None)

    args = parser.parse_args()
    configs = [join(args.configs, f) for f in listdir(args.configs) if isfile(join(args.configs, f))]

    for config in configs:
        params = read_config(config)
        try:
            params_new = override_params(params, args.params)
            write_config(params_new, config)
        except AttributeError:
            print("Cannot override config file ", config)
