from numpy.random import RandomState
from random import Random

seed = 2

py_rng = Random(seed)
np_rng = RandomState(seed)

def set_seed(n):
    global seed, py_rng, np_rng

    seed = n
    py_rng = Random(seed)
    np_rng = RandomState(seed)
