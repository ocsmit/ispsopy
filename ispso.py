import numpy as np
import math
import pandas as pd
import random
from scipy.stats.qmc import Sobol


class ispso_parameters:

    # Double type vector
    f = None

    # Dimension
    D = None

    # Double type vector
    xmin = None

    # Double type vector
    xmax = None

    # D dimension vector, max speed of starting particles
    vmax = None

    # Max speed of new particle, Scalar
    vmax0 = None

    # Swarm size
    S = None

    c1 = 2.05
    c2 = 2.05

    w = 2 / np.abs(2 - c1 - c2 - ((c1 + c2) ** 2 - 4 * (c1 - c2)) ** 0.5)
    rspecies = None
    rprey = None
    rnest = None
    age = None
    xeps = None
    feps = None
    exclusion_factor = None
    maxiter = None

    # Optional
    deterministic = False
    plot_method = "movement"
    plot_x = 1
    plot_delay = 0
    plot_distance_to_solution = 0.05
    plot_save_prefix = ""
    dont_stop = True


# Functions
def f5(x, sol=False):
    if sol:
        return np.array([[0.08], [0.246], [0.449], [0.679], [0.93]])
    else:
        return (
            1
            - math.exp(-2 * math.log(2) * ((x - 0.08) / 0.854) ** 2)
            * math.sin(5 * math.pi * (x ** (3 / 4) - 0.05)) ** 6
        )


def f4(x, sol=False):
    if sol:
        return np.array([[3.0, 2.0], [-3.78, -3.28], [3.58, -1.86], [-2.815, 3.125]])
    if isinstance(x, np.ndarray):
        return (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    else:
        raise Exception("Not type np.ndarray")

def diagonal(s):
    return sum((s.xmax - s.xmin)**2)**0.5

# TESTING PARAMETERS
s = ispso_parameters()
s.f = f4
s.D = 2
s.xmin = np.repeat(-6, s.D)
s.xmax = np.repeat(6, s.D)
s.S = 10 + math.floor(2 * (s.D)**0.5)
s.vmax = (s.xmax - s.xmin) * 0.1
s.exclusion_factor = 3
s.maxiter = 200
s.xeps = 0.001
s.feps = 0.0001
s.rprey = diagonal(s) * 0.0001
s.age = 10
s.rspecies = diagonal(s) * 0.1
s.rnest = diagonal(s) * 0.1
s.plot_distance_to_solution = 0.01
print(s.xmax, s.xmin, s.rprey, s.w, s.deterministic)


# Subroutines
def plotswarm(s):
    na = ["", "profile", "diversity", "mean_diversity", 1]
    if s.plot_method in na:
        return False
    return True


def plotmethod(s, method):
    if method == s.plot_method:
        return True
    return False


def mydist(*args):
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


class ispso:
    def __init__(self, s):
        self.s = s

        if s.S < 2:
            raise Exception("Swarm size must be greater than 1")

        self.best = float("inf")
        self.worst = -self.best

        if s.deterministic:
            if "seed_sobol" not in locals():
                seed_sobol = 4711
            if "seed_random" in locals():
                Random_seed = self.seed_random
            else:
                random.seed(0)
                self.seed_random = [random.uniform(-1, 1) for i in range(626)]
        else:
            seed_sobol = int(random.uniform(0, 1) * 100000)
            seed_random = [random.uniform(-1, 1) for i in range(626)]

        pop = pd.DataFrame()
        if pop.empty:
            self.x = self.new_x(s.S, seed_sobol)
        else:
            x = pop.sort_values("f", ascending=False)
            self.x = x[x.columns[0: self.s.get("D")]].iloc[0: self.s.get("S")]

        self.pbest = np.zeros([s.S, s.D + 1])
        self.gb = 0
        self.gbest = np.zeros([s.D + 1])
        self.gbest[s.D] = self.best
        self.pbest[:, [s.D]] = self.best

        self.prev_gbestf = self.worst
        self.f = []
        self.age = np.repeat(0, s.S)
        self.V = []
        self.halflife_age = myround(0.5 * s.age)

        self.seed = []
        self.species = []

    def start(self):
        diversity = mean_diversity = []
        evals = iter = 0
        num_exclusions_per_nest = []
        num_exclusions = 0

        while True:
            iter += 1
            diversity[iter] = np.mean(np.sqrt((self.x.T -
                np.amin(self.x, axis=0)).T))
            mean_diversity[iter] = np.mean(diversity[:iter])

            self.evaluate_f()

    def new_x(self, n=1, seed=-1):
        """
        New particle positions
        """
        if seed >= 0:
            r = Sobol(self.s.D, True, seed).random(n)
        else:
            r = Sobol(self.s.D, True).random(n)
        return r

    def new_v(self, n=1):
        """
        New particle velocities
        """
        v = pd.DataFrame()
        for i in range(n):
            r = random.uniform(0, 1)
            v[i] = [self.s.vmax0 / np.sqrt(np.sum(r ** 2)) * r]
        return v.T

    def evaluate_f(self, s):

        f = []
        for i in range(self.S):
            f[i] = s.F(self.x[i])
            if f[i] < self.pbest[i, self.s.D] or (f[i] == float("inf") and
                    self.pbest[i, self.s.D] == float("inf")):
                self.pbest[i] = [self.x[i, ], f[i]]
                if f[i] < self.gbest[self.s.D] or (f[i] == float("inf") and
                        self.gbest[self.s.get('D')] == float("inf")):
                    self.gbest = [self.x[i, ], f[i]]
                    self.gb = 1

        return f

    def update_v(self, f):
        lbest = np.zeros([self.s.S, self.s.D + 1])
        l = sorted(f)
        species = []
        self.seed = []
        isolated = np.repeat(0, self.s.S)
        for i in self.s.S:
            if self.seed == None:
                lbest[l[i], ] = self.x[l[i],]
                self.seed = l[i]

        return
