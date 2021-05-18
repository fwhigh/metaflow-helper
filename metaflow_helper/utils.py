from typing import Union
from importlib import import_module
from sklearn.model_selection import ParameterGrid, ParameterSampler


def import_object_from_string(path):
    path, obj_str = path.rsplit('.', 1)
    module_ = import_module(path)
    obj = getattr(module_, obj_str)
    return obj


def explode_parameters(p: Union[tuple, list, dict]):
    if isinstance(p, dict):
        result = _explode_parameter_single_dict(p)
    else:
        result = []
        for item in p:
            result.extend(_explode_parameter_single_dict(item))
    return result


def _explode_parameter_single_dict(p: dict):
    if '__n_iter' in p:
        result = ParameterSampler(
            {k: v for k, v in p.items() if k != '__n_iter'},
            p['__n_iter'],
        )
    else:
        result = ParameterGrid(p)
    return result
