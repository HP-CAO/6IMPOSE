import os
import json


def read_lines(p):
    with open(p, 'r') as f:
        return [
            line.strip() for line in f.readlines()
        ]


def read_json(capture_json_path):
    print('Parse Params file here from ', capture_json_path, ' and pass into main')
    json_data = open(capture_json_path, "r").read()
    # return json.loads(json_data, object_hook=lambda d: Namespace(**d))
    return json.loads(json_data)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


class CachedProperty(object):
    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, inst, cls):
        if inst is None:
            return self
        elif self.name in inst.__dict__:
            return inst.__dict__[self.name]
        else:
            result = self.method(inst)
            inst.__dict__[self.name] = result
            return result

    def __set__(self, inst, value):
        raise AttributeError("This property is read-only")

    def __delete__(self, inst):
        del inst.__dict__[self.name]
