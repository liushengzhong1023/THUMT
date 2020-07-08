# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def load_vocabulary(filename):
    '''
    Load the vocabulary file into a list. Return the vocabulary list
    '''
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip()
            vocab.append(word)

    return vocab


def process_vocabulary(vocab, params):
    '''
    Add the eos into the vocabulary if exist.
    '''
    if params.append_eos:
        vocab.append(params.eos)

    return vocab


def get_control_mapping(vocab, symbols):
    '''
    Symbols are all control symbols. Get the mapping of {control symbol: index in vocabulary}
    '''
    mapping = {}

    for i, token in enumerate(vocab):
        for symbol in symbols:
            if symbol == token:
                mapping[symbol] = i

    return mapping
