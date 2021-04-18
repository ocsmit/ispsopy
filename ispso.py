import numpy as np
import random
from scipy.stats.qmc import Sobol


# Subroutines
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


class ispso:
    def __init__(self, s):

        if s.get("deterministic") is None:
            s["deterministic"] = False
        if s.get("stop_after_solutions") is None:
            s["stop_after_solutions"] = 0
        if s.get("plot_distance_to_solution") is None:
            s["plot_distance_to_solution"] = 0.05
        if s.get("plot_method") is None:
            s["plot_method"] = "movement"
        if s.get("plot_x") is None:
            s["plot_x"] = [1, 2]
        if s.get("plot_delay") is None:
            s["plot_delay"] = 0
        if s.get("plot_save_prefix") is None:
            s["plot_save_prefix"] = ""

        if s.get("S") < 2:
            raise Exception("Swarm size must be greater than 1")

        if s.get("plot_save_prefix") == "":
            _plot_save_format = ""
        else:
            _plot_save_format = (
                f"{s.get('plot_save_prefix')}"
                "{np.floor(np.log10{s.get('maxiter')+1)}"
            )

        if s.get("dont_stop") is None:
            s["dont_stop"] = False

        if s.get("c1") is None:
            s["c1"] = 2.05
        if s.get("c2") is None:
            s["c2"] = 2.05
        if s.get("w") is None:
            s["w"] = 2 / np.abs(
                2
                - s.get("c1")
                - s.get("c2")
                - np.sqrt(
                    (s.get("c1") + s.get("c2")) ** 2 - 4 *
                    (s.get("c1") + s.get("c2"))
                )
            )

        self.s = s

        self.best = float("inf")
        self.worst = -self.best

        if s.get('deterministic'):
            if 'seed_sobol' not in locals():
                seed_sobol = 4711
            if 'seed_random' in locals():
                Random_seed = self.seed_random
            else:
                random.seed(0)
                self.seed_random = [random.uniform(-1, 1)
                                    for i in range(626)]
        else:
            seed_sobol = int(random.uniform(0, 1)*100000)
            seed_random = [random.uniform(-1, 1)
                           for i in range(626)]

        pop = []
        if len(pop) == 0:
            x = self.new_x(s.get("S"), seed_sobol)



        self.pbest = np.zeros([s.get("S"), s.get("D") + 1])
        self.gb = 0
        self.gbest = np.zeros([s.get("D") + 1])
        self.gbest[s.get("D")] = self.best
        self.pbest[:, [s.get("D")]] = self.best

        self.prev_gbestf = self.worst
        self.f = []
        self.age = np.repeat(0, s.get("S"))
        self.V = []
        self.halflife_age = myround(0.5 * s.get("age"))

        self.seed = []
        self.species = []

        pop = []
        if pop is None:
            x = self.new_x(s.get("S"), seed.get("sobol"))

    def start(self):
        diversity = mean_diversity = []
        evals = iter = 0
        num_exclusions_per_nest = []
        num_exclusions = 0

        # while True:
        #    iter += 1
        #    diversity[iter] = np.mean(np.sqrt((x.T -  np.amin(x, axis=0)).T))

    def new_x(self, n=1, seed=-1):
        if seed >= 0:
            r = Sobol(self.s.get("D"), True, seed).random(n)
        else:
            r = Sobol(self.s.get("D"), True).random(n)

        return r

    def evaluate_f(self, s):

        f = []
        for i in range(1, self.S):
            f[i] = s.get("F")

        return
