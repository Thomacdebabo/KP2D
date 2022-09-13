# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""Logging utilities for training
"""

from termcolor import colored

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print ('func:%r  took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

def printcolor_single(message, color="white"):
    """Print a message in a certain color"""
    print(colored(message, color))


def printcolor(message, color="white"):
    "Print a message in a certain color (only rank 0)"

    print(colored(message, color))
