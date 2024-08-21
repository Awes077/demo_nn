import numpy as np
from e3_data_class import *
from dataclasses import dataclass
from typing import List



if len(sys.argv) != 2:
    print("Usage ", sys.argv[0]," <infile> ")
    sys.exit()
else:
    pars_file = sys.argv[1]


path = os.getcwd()
test_pop = PopGenTraining_h(
    seed=600,
    N_reps=20000,
    sample_size=20,
    training_fractions=[0.8, 0.1, 0.1],
    n_cpu=1,
    prefix=path
)


test_pop.generate_training_data(pars_file)

