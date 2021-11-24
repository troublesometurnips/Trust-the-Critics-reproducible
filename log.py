"""Code initially taken from github.com/caogang/wgan-gp"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():#index incrementer
    _iter[0] += 1

def offset(val):#used when loading from checkpoint
    _iter[0] += val 

def plot(name, value):#record data
    _since_last_flush[name][_iter[0]] = value

def flush(temp_save):#save and print recent averages of data
    prints = []
    
    for name, vals in _since_last_flush.items():
        prints.append("{:.{prec1}}\t{:.{prec2}f}".format(name, np.mean(list(vals.values())),prec1 = 5, prec2 = 3))
        _since_beginning[name].update(vals)

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    with open(temp_save + '/log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
