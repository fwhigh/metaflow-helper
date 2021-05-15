from enum import Enum


class RunMode(str, Enum):
    TEST = 'TEST'
    TRAIN = 'TRAIN'
    PREDICT = 'PREDICT'
