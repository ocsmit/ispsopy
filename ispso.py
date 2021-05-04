###############################################################################
# Name:    ispso
# Purpose: Python port of ISPSO.
#          Original R code can be found at github.com/HuidaeCho/ispso.git
# Author:  Owen Smith
# Since:   2021-05-03
###############################################################################
import numpy as np
import sys
import pandas as pd
import random
from scipy.stats.qmc import Sobol

np.seterr(divide="ignore")
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class ispso_parameters:
    '''
    Wrapper for ISPSO parameters

    Attributes
    ----------
        f : function
            Double type vector
            default: None
        D : integer
            Dimension
            default: None
        xmin : np.ndarray
            Double type vector
            default: None
        xmax : np.ndarray
            Double type vector
            default: None
        vmax : np.ndarray
            D dimension vector, max speed of starting particles
            default: None
        vmax0 : float
            Max speed of new particle, Scalar
            default: None
        S : integer
            Swarm size
            default: None
        rspecies : float
            Species radius
            default: None
        rprey : float
            Prey radius
            default: None
        rnest : float
            Nest radius
            default: None
        age : integer
            Age threshold
            default: None
        xeps : float
            Threshold value for normalized geometric mean
            default: None
        feps : float
            Threshold value for the standard deviation of a particles past
            halflife fitness values
            default: None
        exclusion_factor : integer
            Exclusion factor
            default: None
        maxiter : integer
            Maximum number of iterations
            default: None
        c1 : float
            Cognitive coefficient
            default: 2.05
        c2 : float
            Social coefficient
            default: 2.05
        w : float
            Constriction factor
            default: (2 / np.abs(2 - c1 - c2 - ((c1 + c2) ** 2 - 4 *
                      (c1 - c2)) ** 0.5)
        deterministic : boolean
            default: False
        dont_stop : boolean
            Whether or not to stop when max iter is reached
            default: True
        stop_after_solutions : integer
            Number of solutions to stop after. Only stopped if
            stop_after_solutions > 0.
            default: 0
    '''

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

    # Species radius
    rspecies = None

    # Prey radius
    rprey = None

    # Nest radius
    rnest = None

    # Age threshold
    age = None

    # Threshold value for normalized geometric mean
    xeps = None

    # Threshold value for the standard deviation of a particles past
    # halflife fitness values
    feps = None

    # Exclusion factor
    exclusion_factor = None

    # Maximum number of iterations
    maxiter = None

    # Optional
    # Cognitive coefficient
    c1 = 2.05
    # Social coefficient
    c2 = 2.05

    # Constriction factor
    w = 2 / np.abs(2 - c1 - c2 - ((c1 + c2) ** 2 - 4 * (c1 - c2)) ** 0.5)

    # Currently unused
    deterministic = False

    # Misc
    dont_stop = True
    stop_after_solutions = 0

# Subroutines
def diagonal(s):
    return sum((s.xmax - s.xmin) ** 2) ** 0.5


def mydist(*args):
    '''
    Create two dimensional difference array for input variables.

    Values can be passed as either multiple variables, a list, or both a
    variable and a list. If both a variable and a list is passed, then it
    should be mydist(value, list).

    '''
    # Define function for which takes only list
    def dist(*argList):
        X, Y = np.meshgrid(argList, argList)
        return np.abs(X - Y)
    # if len of args is 1 then it is only a list
    if len(args) == 1:
        return dist(args)
    # if args are (variable, list) then insert variable to beginning
    # of list and pass to dist
    if type(args[1]) == list or np.ndarray:
        ll = list(args[1])
        ll.insert(0, args[0])
        return dist(ll)
    # Else if just multiple variables create meshgrid
    else:
        X, Y = np.meshgrid(args, args)
        return np.abs(X - Y)


def mydist2(arr):
    '''
    Calculate euclidean distance of array
    '''
    return np.sqrt(np.sum(np.square(arr)))


def colmax(x):
    '''
    Get max of column
    '''
    return np.amax(x, axis=0)


def colmin(x):
    '''
    Get min of column
    '''
    return np.amin(x, axis=0)


def myround(x, digits=0):
    '''
    Rounding function
    '''
    return np.around(x, digits)


def arr_assign(arr, key, val):
    '''
    Provide dynamic array extending like that in R, Perl, Ruby, etc..
    '''
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
        '''
        Isolated Speciation based Particle Swarm Optimization
        '''

        # Assign parameter class
        self.s = s

        # Create empty nest with required column names
        self.nest = pd.DataFrame(columns=["x", "f", "v", "age", "run", "evals"])

        # Particle swarm must be greater than 1
        if s.S < 2:
            raise Exception("Swarm size must be greater than 1")

        # Assign best and worst as inf, -inf
        self.best = float("inf")
        self.worst = -self.best

        # Assign s
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

        # Create empty pop dataframe
        self.pop = pd.DataFrame(columns=["x", "f", "v", "age"])

        # if pop dataframe is empty then create new coordinates
        if self.pop.empty:
            self.x = self.new_x(s.S, seed_sobol)
        else:
            x = self.pop.sort_values("f", ascending=False)
            self.x = x[x.columns[0 : self.s.get("D")]].iloc[0 : self.s.get("S")]

        # Initialize starting velocities
        self.v = self.new_v(s.S)

        # initialize pbest as 2d array of zeros
        self.pbest = np.zeros([s.S, s.D + 1])

        # initialize gb as 0
        self.gb = 0

        # initialize gbest as 2d array of zeros
        self.gbest = np.zeros([s.D + 1])

        # Assign second columns of gbest and pbest as inf
        self.gbest[s.D] = self.best
        self.pbest[:, [s.D]] = self.best

        # Create empty vector for function values
        self.f = []

        # Initialize ages for each species as 0
        self.age = np.repeat(0, s.S)

        # Initialize empty vector
        self.V = []

        # Initialize halflife age
        self.halflife_age = int(myround(0.5 * s.age))

        # Empty vectors for seed and species
        self.seed = []
        self.species = []

    def run(self):
        '''
        Run ISPSO
        '''
        # Initialize evaluations and iteration count
        self.evals = self.iter = 0
        # Initialize empty vector for nest exclusions
        self.num_exclusions_per_nest = []
        # Initialize number of exclusions
        self.num_exclusions = 0

        # Repeat search
        while True:
            # Increase iteration
            self.iter += 1
            # Evaluate funtion
            self.evaluate_f()
            # Update velocities
            self.update_v()
            # Check solution criterion
            check = self.check_for_convergence()
            # If check is true or max iteration reached then break loop
            if check or (self.s.maxiter and self.iter == self.s.maxiter):
                break
            # Update coordinates
            self.update_x()

        # Return dictionary
        return {
            "iter": self.iter,
            "evals": self.evals,
            "nest": self.nest,
            "pop": self.pop,
        }

    def update_x(self):

        # Adding coordinate array to velocity dataframe produces a dataframe.
        # convert output to numpy
        self.x = (self.x + self.v).to_numpy()

        # {PREY
        # Inferior particles are preyed by superior ones in neighbours. Note
        # that x and f do not correspond because x has been updated since f was
        # evaluated. Therefore, information sharing is based on the past
        # experiences (pbest and the previous position's f value)

        # Get difference array
        d = mydist(self.x)

        # get number of rows
        n = d.shape[0]

        # Initialize preyed as array of 0s
        preyed = np.repeat(0, self.s.S)

        # Iterate to n - 1, with each iteration going from iteration to n
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                # Skip criterion
                if preyed[j] or d[i, j] > self.s.rprey:
                    continue
                # Else set preyed to 1
                preyed[j] = 1
                # Share pbest info
                if self.pbest[i, self.s.D] > self.pbest[j, self.s.D]:
                    self.pbest[i] = self.pbest[j]
                # share info about previous pattern
                if self.f[i] > self.f[j]:
                    self.x[i] = self.x[j]
                    self.v.iloc[i] = self.v.iloc[j]

                # Update coordinate.
                # Transpose and unpack 2d array into 1d vector
                self.x.T[0][j] = self.new_x()
                self.v.iloc[i] = self.v.iloc[j]
                self.pbest[j] = [self.x[j], self.best]
                self.age[j] = 0
        # end of prey

        # {CHECK_NESTS
        # Check existing nests before flying to new points.

        # If nest is created
        if not self.nest.empty:
            # Get difference array
            d = mydist(np.append(self.nest["x"].to_numpy(), self.x.T[0]))

            # Get rows
            n = self.nest.shape[0]

            for i in range(n):
                # Set rnest
                rnst = self.s.rnest
                for j in np.where(d[i, n : self.s.S] <= rnst)[0]:
                    if j == self.gb:
                        self.gb = np.argsort(self.pbest[:, self.s.D])[1]
                        self.gbest = self.pbest[self.gb]
                    # Increse exclusions
                    self.num_exclusions += 1
                    # Set new velocity
                    self.v.iloc[j] = self.new_v()
                    self.pbest[j] = [self.x[j], self.best]
                    # Reset age
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

    def evaluate_f(self):
        '''
        Evaluate objective function
        '''

        # iterate species
        for i in range(self.s.S):

            # Assign function value of coordinate i to f vector
            arr_assign(self.f, i, self.s.f(self.x[i]))
            if self.f[i] < self.pbest[i, self.s.D] or (
                self.f[i] == float("inf") and self.pbest[i, self.s.D + 1] == float("inf")
            ):
                # Add to pbest
                self.pbest[i] = [self.x[i], self.f[i]]
                if self.f[i] < self.gbest[self.s.D] or (
                    self.f[i] == float("inf") and self.gbest[self.s.get("D")] == float("inf")
                ):
                    # Add to gbest
                    self.gbest = [self.x[i], self.f[i]]
                    self.gb = i

        # increase evaluation count and age
        self.evals += self.s.S
        self.age += 1

    def update_v(self):

        # initialize lbest as 2d array of zeros of dimensions [species, D]
        lbest = np.zeros([self.s.S, self.s.D])

        # Sort indexes of function values in ascending order
        l = np.argsort(self.f)
        # Reset species and seed to empty vectors
        self.species = []
        self.seed = []

        # Initialize isolated vector with ones
        isolated = np.repeat(1, self.s.S)

        # Iterate over species
        for i in range(self.s.S):
            if len(self.seed) == 0:
                # Transpose lbest array to make a 1d vector, unpack the inner
                # array and get assign the coordinate at index li to lbest[li]
                lbest.T[0][l[i]] = self.x[l[i]]

                # Add the first position to the seed
                self.seed.append(l[i])

                # Add the seed to the correct position with in the species vector
                arr_assign(self.species, self.seed[-1], self.seed[-1])
                continue
            n = len(self.seed)
            # Initialize found as False
            found = False
            # iterate over seed
            for j in range(n):
                # Check if euclidean distance between two points are less than
                # rspecies.
                if mydist2(self.x[self.seed[j]] - self.x[l[i]]) <= self.s.rspecies:
                    # Set found to True
                    found = True

                    # Insert index and seed into isolated array
                    np.put(isolated, [l[i], self.seed[j]], 0)

                    # Insert seed at index j to species at index li
                    arr_assign(self.species, l[i], self.seed[j])

                    # Transpose lbest array to make a 1d vector, unpack the inner
                    # array and get assign the coordinate at current seed
                    lbest.T[0][l[i]] = self.x[self.seed[j]]
                    break
            # if found is true, continue to next iteration in the local loop
            if found:
                continue

            # If found is not true
            # Assign index l[i] to seed at index n
            arr_assign(self.seed, n, l[i])

            # Assign end seed to species
            arr_assign(self.species, self.seed[n], self.seed[n])

            # increase n
            n += 1

            # Transpose lbest array to make a 1d vector, unpack the inner
            # array and get assign the coordinate at current seed
            lbest.T[0][l[i]] = self.x[l[i]]

            # set fseed to current function value
            fseed = self.f[l[i]]

            # {SPSO_NEIGHBOURING_SPECIES
            # A new species seed searches for superior particles within its
            # speciation radius that failed to form their own species, but
            # happened to belong to seeds with better fitness values.  This
            # behaviour allows superior particles to share their
            # information with neighbouring species having relatively poor
            # fitness values.

            for j in range(self.s.S):
                if j == l[i]:
                    continue
                if (
                    self.f[j] < fseed
                    and mydist2(self.x[j] - self.x[l[i]]) <= self.s.rspecies
                ):
                    lbest.T[0][l[i]] = self.x[j]
                    fseed = self.f[j]

            # End of SPSO_NEIGHBOURING_SPECIES}

            # {SPSO_NEIGHBOURING_PBESTS
            # A new species seed searches for superior pbests within its
            # speciation radius. This behaviour allows superior pbests to
            # share their information with neighbouring species having
            # relatively poor fitness values.
            for j in range(self.s.S):
                if j == l[i]:
                    continue
                if (
                    self.f[j] < fseed
                    and mydist2(self.pbest[j][self.s.D] - self.x[l[i]])
                    <= self.s.rspecies
                ):
                    lbest.T[0][l[i]] = self.pbest[j][self.s.D]
                    fseed = self.pbest[j][self.s.D]

            # End of SPSO_NEIGHBOURING_PBESTS}

        # {SPSO_ISOLATED_SPECIES
        # Isolated particles form one species.
        if any(isolated == 1):
            # create an empty vector
            seed_l = []

            # populate seed_l with values in self.seed, but dropping all seeds
            # where isolated is equal to zero
            for x in self.seed:
                if isolated[x] == 1:
                    seed_l.append(x)

            # Set seed to new isolated species
            self.seed = seed_l

            # increase age of isolated seeds
            self.age[isolated == 1] += 1
            n = len(seed_l) - 1

            # Create temporary dataframe
            tmp = pd.DataFrame()

            # Add column for isolated values
            tmp["isolated"] = isolated

            # Add function values for isolated particles
            tmp["f"] = self.f

            # Sort based on values
            tmp = tmp.sort_values("f")

            # Make list of indices where isolated == 1
            tmp = tmp[tmp["isolated"] == 1].index.tolist()

            # Set seed at n to first value in tmp
            self.seed[n] = tmp[0]

            # set species at nn to negative seed
            self.species[n] = -self.seed[n]

            # iterate over isolated species
            for ii in np.where(isolated == 1)[0]:
                # Assign coordinate to lbest
                lbest[ii] = self.x[self.seed[n]]
                # assign species negative seed to species
                arr_assign(self.species, ii, -self.seed[n])
        # End of SPSO_ISOLATED_SPECIES}

        # Constriction PSO (Clerc and Kennedy, 2000)
        self.v = (
            self.s.w
            * (
                self.v.T
                + self.s.c1 * np.random.uniform(size=(self.s.D)) * self.pbest[:, 0]
                + self.s.c2 * np.random.uniform(size=(self.s.D)) * lbest.T
            )
        ).T
        # convert pandas dataframe to numpy array to get only values
        v = self.v.to_numpy()

        # Check if nest is empty
        if not self.nest.empty:
            # iterate over seeds
            for i in self.seed:
                # Get 2d difference array
                dist = mydist(self.x[i][0] + v[i], self.nest["x"])
                # Get distance in first row
                dist2 = dist[0, self.nest.shape[0]]
                # Check
                if any(dist2 < 2 * self.s.rnest):
                    # Increase num_exclusions
                    self.num_exclusions += 1
                    # Update velocity in velocity array
                    v[i] = v[i] + self.s.rspecies * np.random.uniform(size=(self.s.D))
        # end check nest

        # if velocity is less than -max, return -max
        v = np.maximum(-self.s.vmax, v)
        # if velocity is greater than +max, return +max
        v = np.minimum(self.s.vmax, v)

        # Confinment
        # iterate over species
        for i in range(self.s.S):
            # return index where x + v is less greater than coordinate max
            # Get first index because np.where returns 2d array
            j = np.where(self.x[i] + v[i] > self.s.xmin)[0]
            k = len(j)
            # if k > 0
            if k:
                v[i, j] = (self.x[i, j] - self.s.xmin[j][0]) * np.random.uniform(
                    size=(k)
                )
            j = np.where(self.x[i] + v[i] > self.s.xmax)[0]
            k = len(j)
            # if k > 0
            if k:
                v[i, j] = (self.s.xmax[j][0] - self.x[i, j]) * np.random.uniform(
                    size=(k)
                )
        # End confinment

        # Convert back to dataframe
        self.v = pd.DataFrame(v)

        # Take sqrt of v^2
        self.V = np.sum(v ** 2, axis=1) ** 0.5

        # Generate pop dataframe with dictionary and append
        tmp = pd.DataFrame(
            {"x": self.x.flatten(), "f": self.f, "v": self.V, "age": self.age}
        )
        self.pop = self.pop.append(tmp, ignore_index=True)

    def fly_away_and_substitute(self, neighbors):
        # Sum of neighbors
        n = np.sum(neighbors)
        # Replace
        self.x[neighbors] = self.new_x(n)
        # Convert to array for assigning new velocity,
        # does not persist as array
        self.v.to_numpy()[neighbors] = self.new_v(n)
        # rest age
        self.age[neighbors] = 0

    def check_for_convergence(self):
        # Nest by age

        # Iterate over seed
        for i in self.seed:
            # if age less than initial age, skip
            if self.age[i] < self.s.age:
                continue
            # Create range of values to pull from the total population
            # List is converted to array to provide effecient way to multiply
            # and add to all values
            index_range = (
                self.s.S
                * np.array([i for i in range(self.iter - self.halflife_age, self.iter)])
                + i
            )
            # Pull out values from population and invert order
            halflife = self.pop.iloc[index_range][::-1]

            # Generate values for evaluation on whether or not to creat nest
            # Convert all dataframes to numpy
            if self.s.D == 1:
                # if dimension is 1 then transpose
                cmax = colmax(halflife.iloc[:, 0 : self.s.D].T.to_numpy())
                cmin = colmin(halflife.iloc[:, 0 : self.s.D].T.to_numpy())
            else:
                cmax = colmax(halflife.iloc[:, 0 : self.s.D].to_numpy())
                cmin = colmin(halflife.iloc[:, 0 : self.s.D].to_numpy())

            # evaluation
            ev = np.exp(np.mean(np.log((cmax - cmin) / (self.s.xmax - self.s.xmin))))

            # compare evaluation
            if ev <= self.s.xeps and np.std(halflife["f"]) <= self.s.feps:
                # Reduce run
                run = self.evals - self.s.S + i
                # Create dataframe
                tmp = pd.DataFrame(
                    {
                        "x": self.x[i],
                        "f": self.f[i],
                        "v": self.V[i],
                        "age": self.age[i],
                        "run": run,
                        "evals": self.evals,
                    }
                )
                # Add to nest
                self.nest = self.nest.append(tmp, ignore_index=True)
                # Update number of exclusions
                arr_assign(
                    self.num_exclusions_per_nest,
                    self.nest.shape[0] - 1,
                    self.num_exclusions,
                )
                self.num_exclusions = 0
                print(
                    f"{self.x[i]} added at iter={self.iter}, run={run} "
                    f"evals={self.evals}, nest={self.nest.shape[0] - 1}"
                )

                # Substitute for new values in found nest
                self.fly_away_and_substitute(
                    mydist(self.x[:, : self.s.D])[i] <= self.s.rspecies
                )

        # Evaluate nest
        if not self.nest.empty and self.s.exclusion_factor and self.num_exclusions:
            # get run column and convert to numpy
            og = self.nest["run"].to_numpy()
            # Create new
            if self.nest.shape[0] == 1:
                # if only one value in array, take it
                new = self.nest["run"].to_numpy()
            else:
                # if more than one value in array, drom the last value
                new = self.nest.drop(index=self.nest.shape[0] - 1)["run"].to_numpy()
            # Add zero to beginning of array
            new = np.insert(new, 0, 0)
            delta_sol_iters = (og - new) / self.s.S
            average_delta_sol_iter = np.mean(delta_sol_iters)
            delta_curr_iter = delta_sol_iters[self.nest.shape[0] - 1]

            # compute func_difficulty
            func_difficulty = np.max(delta_sol_iters) / average_delta_sol_iter

        # Evaluate if found is true
        if (
            not self.s.dont_stop
            and self.num_exclusions / self.s.S
            > self.s.exclusion_factor * func_difficulty
        ):
            return True

        if not self.s.dont_stop:
            if self.s.stop_after_solutions > 0:
                return self.s.stop_after_solutions == self.nest.shape[0]

        return False
