import numpy as np


# Subroutines

def main(s):
    diversity = mean_diversity = []
    evals = iter = 0
    num_exclusions_per_nest = []
    num_exclusions = 0


def plotswarm(s):
    na = ["", "profile", "diversity", "mean_diversity", 1]
    if s.get("plot_method") in na:
        return False
    return True


def plotmethod(s, method):
    if method == s.get("plot_method"):
        return True
    return False


def mydist(s, *args):
    X, Y = np.meshgrid(args, args)
    return np.abs(X - Y)


def mydist2(arr):
    return np.sqrt(np.sum(np.square(arr)))


def mynrow(x):
    if x is None:
        return 0
    elif type(x) in (tuple, list) or len(x.shape) == 1:
        return 1
    else:
        return x.shape[0]


def myncol(x):
    if x is None:
        return 0
    elif type(x) in (tuple, list) or len(x.shape) == 1:
        return 1
    else:
        return x.shape[1]


def colmax(x):
    return np.amax(x, axis=0)


def colmin(x):
    return np.amin(x, axis=0)


def myround(x, digits=0):
    return np.around(x, digits)

