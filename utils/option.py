from collections import OrderedDict
import json
from utils.io import log

def parse(opt_path):
    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    return opt


def recursive_print(src, dpth=0, key=None):
    """ Recursively prints nested elements."""
    tabs = lambda n: ' ' * n * 4 # or 2 or 8 or...

    if isinstance(src, dict):
        if key is not None:
            print(tabs(dpth) + '%s: ' % (key))
        for key, value in src.items():
            recursive_print(value, dpth + 1, key)
    elif isinstance(src, list):
        if key is not None:
            print(tabs(dpth) + '%s: ' % (key))
        for litem in src:
            recursive_print(litem, dpth)
    else:
        if key is not None:
            print(tabs(dpth) + '%s: %s' % (key, src))


def recursive_log(log_file, src, dpth=0, key=None):
    """ Recursively prints nested elements."""
    tabs = lambda n: ' ' * n * 4 # or 2 or 8 or...

    if isinstance(src, dict):
        if key is not None:
            log(log_file, tabs(dpth) + '%s: \n' % (key), with_time=False)
        for key, value in src.items():
            recursive_log(log_file, value, dpth + 1, key)
    elif isinstance(src, list):
        if key is not None:
            log(log_file, tabs(dpth) + '%s: \n' % (key), with_time=False)
        for litem in src:
            recursive_log(log_file, litem, dpth)
    else:
        if key is not None:
            log(log_file, tabs(dpth) + '%s: %s\n' % (key, src), with_time=False)