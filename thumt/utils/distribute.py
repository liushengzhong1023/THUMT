# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

_ENGINE = None


def enable_distributed_training():
    global _ENGINE
    try:
        import horovod.tensorflow as hvd
        _ENGINE = hvd
        hvd.init()
    except ImportError:
        sys.stderr.write("Error: You must install horovod first in order to"
                         " enable distributed training.\n")
        exit()


def is_distributed_training_mode():
    return _ENGINE is not None


def rank():
    '''
    The rank of a process is a unique ID given that distinguishes it from the other processes running your Horovod job
    at the same time.
    Suppose you're running a Horovod job on 2 machines, and each machine has 4 GPUs.
    So in total, you run 8 processes (size = 8). Then the rank will be a number [0, 7],
    and the local rank will be a number [0, 3].
    Return the global ID of current process.
    '''
    return _ENGINE.rank() if _ENGINE is not None else 0


def local_rank():
    '''
    Suppose you're running a Horovod job on 2 machines, and each machine has 4 GPUs.
    So in total, you run 8 processes (size = 8). Then the rank will be a number [0, 7],
    and the local rank will be a number [0, 3].
    Return the local ID of current process.
    '''
    return _ENGINE.local_rank() if _ENGINE is not None else 0


def size():
    '''
    A function that returns the number of Horovod processes.
    '''
    return _ENGINE.size() if _ENGINE is not None else 1


def all_reduce(tensor):
    '''
    AllReduce is an operation that reduces the target arrays in all processes to a single array
        and returns the resultant array to all processes.
    '''
    if _ENGINE is None:
        return tensor

    return _ENGINE.allreduce(tensor, compression=_ENGINE.Compression.fp16)


def get_broadcast_hook():
    '''
    SessionRunHook that will broadcast all global variables from root rank to all other processes during initialization.
    '''
    if not _ENGINE:
        return None
    return _ENGINE.BroadcastGlobalVariablesHook(0)
