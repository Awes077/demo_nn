import numpy as np
from numpy.random import default_rng
import pandas as pd
import msprime
import tskit
import random
import concurrent.futures
import sys
import torch
import gpustat
import os
import matplotlib.pyplot as plt

from dataclasses import dataclass

from e3_data_class import *


path = os.getcwd()
test_pop = PopGenTraining_h(
    seed=6,
    N_reps=20000,
    sample_size=20,
    training_fractions=[0.8, 0.1, 0.1],
    n_cpu=1,
    prefix=path,
)

test_pop.generate_bottle_time()
