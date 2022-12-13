import distutils.util as util
import json
from types import SimpleNamespace as Namespace
import os

from lib.params import NetworkParams


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return setattr_recursive(getattr(obj, split[0]), split[1:], val) if len(split) > 1 else setattr(obj, split[0], val)


def generate_config(params, file_path):
    print("Saving Configs")
    f = open(file_path, "w")
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    print(json_data)
    f.write(json_data)
    f.close()


def write_config(params, config_path):
    f = open(config_path, "w")
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    print(json_data)
    f.write(json_data)
    f.close()


def read_config(config_path) -> NetworkParams:
    print('Parse Params file here from ', config_path, ' and pass into main')
    json_data = open(config_path, "r").read()
    return json.loads(json_data, object_hook=lambda d: Namespace(**d))


def override_params(params: NetworkParams, overrides) -> NetworkParams:
    assert (len(overrides) % 2 == 0)
    for k in range(0, len(overrides), 2):
        oldval = getattr_recursive(params, overrides[k])
        if type(oldval) == bool:
            to_val = bool(util.strtobool(overrides[k + 1]))
        else:
            to_val = type(oldval)(overrides[k + 1])
        setattr_recursive(params, overrides[k], to_val)
        print("Overriding param", overrides[k], "from", oldval, "to", to_val)

    return params


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))
