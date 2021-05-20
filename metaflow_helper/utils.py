import errno
import os
import subprocess
import time
from importlib import import_module
from typing import Union

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


def silent_rm_file(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def system_command_with_retry(cmd: list):
    for i in range(0, 5):
        wait_seconds = 2 ** i
        try:
            status = subprocess.run(cmd)
            if status.returncode != 0:
                print(f'command status was {status}, retrying after {wait_seconds} seconds')
                time.sleep(wait_seconds)
                continue
        except subprocess.CalledProcessError:
            print(f'command failed, retrying after {wait_seconds} seconds')
            time.sleep(wait_seconds)
            continue
        break


def install_dependencies(dependencies: list):
    for dependency in dependencies:
        for k, v in dependency.items():
            try:
                module_ = import_module(k)
            except ModuleNotFoundError:
                system_command_with_retry(['pip', 'install', v])
