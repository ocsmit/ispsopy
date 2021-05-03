from ispso import ispso_parameters, ispso, diagonal
import numpy as np
import math

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


r = ispso(s).run()
print(r.get('nest'))
