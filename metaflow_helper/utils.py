from importlib import import_module


def import_object_from_string(path):
    path, obj_str = path.rsplit('.', 1)
    module_ = import_module(path)
    obj = getattr(module_, obj_str)
    return obj
