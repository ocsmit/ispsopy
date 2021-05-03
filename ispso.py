import numpy as np
import sys
import math
import pandas as pd
import random
from scipy.stats.qmc import Sobol
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

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
    stop_after_solutions = 0


# Functions
def f4(x, sol=False):
    if sol:
        return np.array([[0.08], [0.246], [0.449], [0.679], [0.93]])
    else:
        return (
            1
            - math.exp(-2 * math.log(2) * ((x - 0.08) / 0.854) ** 2)
            * math.sin(5 * math.pi * (x ** (3 / 4) - 0.05)) ** 6
        )


def diagonal(s):
    return sum((s.xmax - s.xmin)**2)**0.5


# TESTING PARAMETERS
s = ispso_parameters()
s.f = f4
s.D = 1
s.xmin = np.array([0])
s.xmax = np.array([1])
s.S = 10 + math.floor(2 * (s.D)**0.5)
s.vmax = (s.xmax - s.xmin) * 0.1
s.vmax0 = diagonal(s)*0.001
s.exclusion_factor = 3
s.maxiter = 200
s.xeps = 0.001
s.feps = 0.0001
s.rprey = diagonal(s) * 0.0001
s.age = 10
s.rspecies = diagonal(s) * 0.1
s.rnest = diagonal(s) * 0.1
s.plot_distance_to_solution = 0.01


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
    def dist(*argList):
        X, Y = np.meshgrid(argList, argList)
        return np.abs(X - Y)
    if len(args) == 1:
        return dist(args)
    if type(args[1]) == list or np.ndarray:
        ll = list(args[1])
        ll.insert(0, args[0])
        return dist(ll)
    else:
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


def arr_assign(arr, key, val):
    try:
        arr[key] = val
        return
    except IndexError:
        assert key >= 0
        arr.extend(((key + 1) - len(arr)) * [None])
        arr[key] = val
        return


class ispso:
    def __init__(self, s):
        self.s = s
        self.nest = pd.DataFrame(columns=['x', 'f', 'v', 'age',
                                         'run', 'evals'])
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

        self.pop = pd.DataFrame(columns=['x', 'f', 'v', 'age'])
        if self.pop.empty:
            self.x = self.new_x(s.S, seed_sobol)
        else:
            x = self.pop.sort_values("f", ascending=False)
            self.x = x[x.columns[0: self.s.get("D")]].iloc[0: self.s.get("S")]
        self.v = self.new_v(s.S)


        self.pbest = np.zeros([s.S, s.D + 1])
        self.gb = 0
        self.gbest = np.zeros([s.D + 1])
        self.gbest[s.D] = self.best
        self.pbest[:, [s.D]] = self.best

        self.prev_gbestf = self.worst
        self.f = []
        self.age = np.repeat(0, s.S)
        self.V = []
        self.halflife_age = int(myround(0.5 * s.age))

        self.seed = []
        self.species = []

    def start(self):
        diversity = mean_diversity = []
        self.evals = self.iter = 0
        self.num_exclusions_per_nest = []
        self.num_exclusions = 0
        while True:
            self.iter += 1
            diversity.append(np.mean(np.sqrt((self.x -np.amin(self.x, axis=0)).T)))
            mean_diversity.append(np.mean(diversity))
            self.evaluate_f(self.s)
            self.update_v()
            check = self.check_for_convergence()
            if check or (self.s.maxiter and self.iter == self.s.maxiter):
                break
            self.update_x()

    def update_x(self):
        self.x = (self.x + self.v).to_numpy()

        #{PREY
        # Inferior particles are preyed by superior ones in neighbours. Note
        # that x and f do not correspond because x has been updated since f was
        # evaluated. Therefore, information sharing is based on the past
        # experiences (pbest and the previous position's f value)
        d = mydist(self.x)
        n = d.shape[0]
        preyed = np.repeat(0, self.s.S)
        for i in range(0, n-1):
            for j in range(i + 1, n):
                if preyed[j] or d[i, j] > self.s.rprey:
                    continue
                preyed[j] = 1
                # Share pbest info
                if self.pbest[i, self.s.D] > self.pbest[j, self.s.D]:
                    self.pbest[i] = self.pbest[j]
                # share info about previous pattern
                if self.f[i] > self.f[j]:
                    self.x[i] = self.x[j]
                    self.v.iloc[i] = self.v.iloc[j]

                self.x.T[0][j] = self.new_x()
                self.v.iloc[i] = self.v.iloc[j]
                self.pbest[j] = [self.x[j], self.best]
                self.age[j - 1] = 0
        # end of prey

        #{CHECK_NESTS
        # Check existing nests before flying to new points.
        if not self.nest.empty:
            d = mydist(np.append(self.nest['x'].to_numpy(), self.x.T[0]))
            n = self.nest.shape[0]

            for i in range(n):
                rnst = self.s.rnest
                for j in np.where(d[i, n:self.s.S] <= rnst)[0]:
                    if j == self.gb:
                        self.gb = np.argsort(self.pbest[:, self.s.D])[1]
                        self.gbest = self.pbest[self.gb]
                    self.num_exclusions += 1
                    self.v.iloc[j] = self.new_v()
                    self.pbest[j] = [self.x[j], self.best]
                    self.age[j] = 0

    def new_x(self, n=1, seed=-1):
        """
        New particle positions
        """
        if seed >= 0:
            r = Sobol(self.s.D, True, seed).random(n)
        else:
            r = Sobol(self.s.D, True).random(n)
        return (self.s.xmin + (self.s.xmax - self.s.xmin) * r.T).T

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
        for i in range(s.S):
            arr_assign(f, i, s.f(self.x[i]))
            if f[i] < self.pbest[i, self.s.D] or (f[i] == float("inf") and
                    self.pbest[i, self.s.D + 1] == float("inf")):
                self.pbest[i] = [self.x[i, ], f[i]]
                if f[i] < self.gbest[self.s.D] or (f[i] == float("inf") and
                        self.gbest[self.s.get('D')] == float("inf")):
                    self.gbest = [self.x[i, ], f[i]]
                    self.gb = 1
        self.evals += self.s.S
        self.age += 1

        self.f = f

    def update_v(self):
        f = self.f
        lbest = np.zeros([self.s.S, self.s.D])
        l = np.argsort(f)
        self.species = []
        self.seed = []
        isolated = np.repeat(1, self.s.S)
        for i in range(self.s.S):
            if len(self.seed) == 0:
                lbest.T[0][l[i]] = self.x[l[i]]
                self.seed.append(l[i])
                arr_assign(self.species,  self.seed[-1], self.seed[-1])
                continue
            n = len(self.seed)
            found = False
            for j in range(n):
                if mydist2(self.x[self.seed[j]] -
                        self.x[l[i]]) <= self.s.rspecies:
                    found = True
                    np.put(isolated, [l[i], self.seed[j]], 0)
                    arr_assign(self.species, l[i], self.seed[j])
                    lbest.T[0][l[i]] = self.x[self.seed[j]]
                    break
            if found:
                continue
            arr_assign(self.seed, n, l[i])
            arr_assign(self.species, self.seed[n], self.seed[n])
            n += 1

            lbest.T[0][l[i]] = self.x[l[i]]

            fseed = f[l[i]]

            #{SPSO_NEIGHBOURING_SPECIES
            # A new species seed searches for superior particles within its
            # speciation radius that failed to form their own species, but
            # happened to belong to seeds with better fitness values.  This
            # behaviour allows superior particles to share their
            # information with neighbouring species having relatively poor
            # fitness values.

            for j in range(self.s.S):
                if j == l[i]:
                    continue
                if f[j] < fseed and mydist2(self.x[j]-self.x[l[i]]) <= self.s.rspecies:
                    lbest.T[0][l[i]] = self.x[j]
                    fseed = f[j]

            # End of SPSO_NEIGHBOURING_SPECIES}

            #{SPSO_NEIGHBOURING_PBESTS
            # A new species seed searches for superior pbests within its
            # speciation radius. This behaviour allows superior pbests to
            # share their information with neighbouring species having
            # relatively poor fitness values.
            for j in range(self.s.S):
                if j == l[i]:
                    continue
                if f[j] < fseed and mydist2(self.pbest[j][self.s.D]-self.x[l[i]]) <= self.s.rspecies:
                    lbest.T[0][l[i]] = self.pbest[j][self.s.D]
                    fseed = self.pbest[j][self.s.D]

            # End of SPSO_NEIGHBOURING_PBESTS}

        #{SPSO_ISOLATED_SPECIES
        # Isolated particles form one species.
        if any(isolated == 1):
            seed_l = []

            for x in self.seed:
                if isolated[x] == 0:
                    seed_l.append(x)
            self.seed = np.array(seed_l)
            self.age[isolated == 1] += 1
            n = len(seed_l) - 1
            tmp = pd.DataFrame()
            tmp["isolated"] = isolated
            tmp["f"] = f
            tmp = tmp.sort_values("f")
            tmp = tmp[tmp["isolated"] == 1].index.tolist()

            self.seed[n] = tmp[0]
            self.species[n] = -self.seed[n]
            for ii in np.where(isolated == 1)[0]:
                lbest[ii] = self.x[self.seed[n]]
                arr_assign(self.species, ii, -self.seed[n])
                self.species[ii] = -self.seed[n]
        # End of SPSO_ISOLATED_SPECIES}

        # Constriction PSO (Clerc and Kennedy, 2000)
        self.v = (
                self.s.w * (self.v.T + self.s.c1 * Sobol(self.s.D).random() *
                        self.pbest[:, 0] + self.s.c2 *
                        Sobol(self.s.D).random() * lbest.T)
                ).T

        v = self.v.to_numpy()
        # Add x_i to v_i
        # get first column to s.D col in nest
        # mydist2(s, add, fcol s.d)[first row, second col:nrow(nest)+1)
        if not self.nest.empty:
            for i in self.seed:
                dist = mydist(self.x[i][0] + v[i], self.nest['x'])
                dist2 = dist[0, self.nest.shape[0]]
                if any(dist2 < 2 * self.s.rnest):
                    self.num_exclusions += 1
                    v[i] = v[i] + self.s.rspecies * Sobol(self.s.D).random()

        v = np.maximum(-self.s.vmax, v)
        v = np.minimum(self.s.vmax, v)

        # Confinment
        for i in range(self.s.S):
            j = np.where(self.x[i] + v[i] > self.s.xmin)[0]
            k = len(j)
            if k:
                v[i, j] = (self.x[i, j] -
                           self.s.xmin[j][0]) * Sobol(k).random()
            j = np.where(self.x[i] + v[i] > self.s.xmax)[0]
            k = len(j)
            if k:
                v[i, j] = (self.s.xmax[j][0] -
                           self.x[i, j]) * Sobol(k).random()
        # End confinment

        self.v = pd.DataFrame(v)

        self.V = np.sum(v**2, axis=1)**0.5

        self.f = f
        tmp = pd.DataFrame({'x': self.x.flatten(),
                            'f': f,
                            'v': self.V,
                            'age': self.age})
        self.pop = self.pop.append(tmp, ignore_index=True)

    def fly_away_and_substitute(self, neighbors):
        n = np.sum(neighbors)
        self.x[neighbors] = self.new_x(n)
        self.v.to_numpy()[neighbors] = self.new_v(n)
        self.age[neighbors] = 0

    def check_for_convergence(self):
        # Nest by age
        for i in self.seed:
            if self.age[i] < self.s.age:
                continue
            halflife = self.pop.iloc[self.s.S * self.iter-1 - self.halflife_age: self.s.S * self.iter-1]
            if self.s.D == 1:
                cmax = colmax(halflife.iloc[:, 0:self.s.D].T.to_numpy())
                cmin = colmin(halflife.iloc[:, 0:self.s.D].T.to_numpy())
            else:
                cmax = colmax(halflife.iloc[:, 0:self.s.D].to_numpy())
                cmin = colmin(halflife.iloc[:, 0:self.s.D].to_numpy())

            ev = np.exp(np.mean(np.log((cmax - cmin) /
                        (self.s.xmax - self.s.xmin))))
            if ev <= self.s.xeps and np.std(halflife['f']) <= self.s.feps:
                run = self.evals - self.s.S + i
                tmp = pd.DataFrame({'x': self.x[i], 'f': self.f[i],
                                    'v': self.V[i], 'age': self.age[i],
                                    'run': run, 'evals': self.evals})
                self.nest = self.nest.append(tmp, ignore_index=True)
                arr_assign(self.num_exclusions_per_nest,
                           self.nest.shape[0] - 1, self.num_exclusions)
                self.num_exclusions = 0

                self.fly_away_and_substitute(mydist(self.x[:,:self.s.D])[i] <= self.s.rspecies)
                print(f"{self.x[i]} added at iter={self.iter}, run={run} "
                      f"evals={self.evals}, nest={self.nest.shape[0] - 1}")


        if not self.nest.empty and self.s.exclusion_factor and self.num_exclusions:
            og = self.nest['run'].to_numpy()
            if self.nest.shape[0] == 1:
                new = self.nest['run'].to_numpy()
            else:
                new = self.nest.drop(index=self.nest.shape[0] - 1)['run'].to_numpy()
            new = np.insert(new, 0, 0)
            delta_sol_iters = (og - new) / self.s.S
            average_delta_sol_iter = np.mean(delta_sol_iters)
            delta_curr_iter = delta_sol_iters[self.nest.shape[0] - 1]

            func_difficulty = np.max(delta_sol_iters) / average_delta_sol_iter

        if not self.s.dont_stop and self.num_exclusions / \
                self.s.S > self.s.exclusion_factor * func_difficulty:
            return True

        if not self.s.dont_stop:
            if self.s.stop_after_solutions > 0:
                return self.s.stop_after_solutions == mynrow(self.nest)

        return False


r = ispso(s)
r.start()
