import argparse
import glob
import json
import numpy as np
import os
import random
    

def manhattan_dist(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.abs(x2 - x1) + np.abs(y2 - y1)

def get_parser():
    """Return a nicely formatted argument parser

    This function is a simple wrapper for the argument parser I like to use,
    which has a stupidly long argument that I always forget.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def load_experiment(tag, coefs=None):
    logfiles = sorted(glob.glob(os.path.join('results/logs', tag + '*', 'train-*.txt')))
    seeds = [f.split('-')[-1].split('.')[0] for f in logfiles]
    logs = [open(f, 'r').read().splitlines() for f in logfiles]

    def read_log(log, coefs=coefs):
        results = [json.loads(item) for item in log]
        fields = results[0].keys()
        data = dict([(f, np.asarray([item[f] for item in results])) for f in fields])
        if coefs is None:
            coefs = {
                'L_inv': 1.0,
                'L_fwd': 0.1,
                'L_cpc': 1.0,
                'L_fac': 0.1,
            }
        if 'L' not in fields:
            data['L'] = sum([
                coefs[f] * data[f] if f != 'L_fac' else coefs[f] * (data[f] - 1)
                for f in coefs.keys()
            ])
        return data

    results = [read_log(log) for log in logs]
    data = dict(zip(seeds, results))
    return data

def get_good_color(color):
    colorname = color
    colorname = 'gold' if colorname == 'yellow' else colorname
    colorname = 'c' if colorname == 'cyan' else colorname
    colorname = 'm' if colorname == 'magenta' else colorname
    colorname = 'silver' if colorname in ['gray', 'grey'] else colorname
    return colorname
