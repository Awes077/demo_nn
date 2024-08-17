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

if len(sys.argv) != 2:
    print("Usage ", sys.argv[0]," <infile> ")
    sys.exit()
else:
    #p = int(sys.argv[1])
    pars_file = sys.argv[1]
    #outfile = sys.argv[2]

#print(p)

def get_idle_gpu():
    """
    Utility function which uses the gpustat module to select the
    least busy GPU that is available and then sets the
    CUDA_VISIBLE_DEVICES environment variable so that
    only that GPU is used
    """

    #Try let's you test a bit of code if there is no error
    try:
        #
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry["index"]), stats)
        ratios = map(
            lambda gpu: float(gpu.entry["memory.used"])
            / float(gpu.entry["memory.total"]),
            stats,
        )
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        return bestGPU
    #except gets run if there is an error, so presumably if gpustat etc doesn't work
    except Exception:
        pass

#this device variable will get used later during training
device = torch.device(f"cuda:{get_idle_gpu()}" if torch.cuda.is_available() else "cpu")

from dataclasses import dataclass
from typing import List


@dataclass
class PopGenTraining_h():
    '''
    
    :param int seed: random seed
    :param int N_reps: number of replicates to simulate
    :param int sample_size: sample size of each simulation
    :param int t_mod: time of change from modern population size to middle epoch population size
    :param int t_mid: time of change from middle epoch population size to ancestral population size
    :param int modern_size: modern effective population size
    :param int mid_size: middle-epoch effective population size
    :param int ancestral_size: ancestral effective population size
    :param int n_loci: number of independent SNP loci to simulate
    :param float theta: theta
    :param str prefix: prefix for the output files -- full path is encouraged
    :param List[float] training_fractions: fractions of the data to use for training, validation, and testing'''


    seed: int
    N_reps: int
    sample_size: int
    training_fractions: list[float]
    prefix: str
    n_cpu: int = 1

    
    # post init is just a set of things called after a data object has been initialized
    def __post_init__(self):
        #check that our test/train/validation split sums to 1
        if sum(self.training_fractions) != 1:
            raise ValueError("training_fractions must sum to 1")
        training_fraction, validation_fraction, test_fraction = self.training_fractions
        # list of labels for each simulation
        self.training_labels = ["training"] * int(training_fraction * self.N_reps) + \
                               ["validation"] * int(validation_fraction * self.N_reps) + \
                               ["test"] * int(test_fraction * self.N_reps)
        # if we don't have the right number of training labels for the number of replications then we throw an error. not sure how exactly that would work but hey. maybe an
        #issue with integer maths?
        #if len(self.hyp_labels) != self.N_reps:
        #    raise ValueError('wrong number of hypothesis labels')
        if len(self.training_labels) != self.N_reps:
            raise ValueError("wrong number of training labels. Check training_fractions")
        # seed the random number generator
        self.rng = default_rng(self.seed)
        # seed for each simulation
        #self.seeds = self.rng.integers(0, 2**32, size = self.N_reps)
        self.ancestral_size = np.zeros(self.N_reps)
        self.mid_size = np.zeros(self.N_reps)
        self.modern_size = np.zeros(self.N_reps)
        self.t_mod = np.zeros(self.N_reps)
        self.t_mid = np.zeros(self.N_reps)
        self.num_loci = np.zeros(self.N_reps)



    def generate_bottle_time(self):
            self.ancestral_size = self.rng.uniform(500,500000, size=self.N_reps).round(3)
            self.mid_size = self.rng.uniform(500, 500000, size=self.N_reps).round(3)
            self.modern_size = self.rng.uniform(500, 500000, size=self.N_reps).round(3)
            #bott_sizes = np.zeros(self.N_reps)
            #pop_sizes = [self.ancestral_size, self.bottle_size, self.modern_size]
            #for i in range(self.N_reps):
            #    bott_sizes[i] =  self.rng.uniform(500, self.ancestral_size[i], size=1).round(3)
            #    print(largest_pop[i])
            self.t_mod = self.rng.uniform(100,4*self.modern_size, size=self.N_reps).round(3)
            self.t_mid = self.rng.uniform(0, 4*self.mid_size, size=self.N_reps).round(3)
            self.num_loci = self.rng.uniform(20000, 100000, size=self.N_reps).astype(int)
            file_paths = [f"{self.prefix}/Empirical_run/Marginal_locus_run/SFS_3e/SFS_T_{self.t_mod[rep]}_L_{self.t_mid[rep]}_A_{self.ancestral_size[rep]}_B_{self.mid_size[rep]}_M_{self.modern_size[rep]}_nloc_{self.num_loci[rep]}.csv" for rep in range(len(self.t_mod))]
            local_paths = [f"/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/Benchmarking/Empirical_run/Marginal_locus_run/SFS_3e/SFS_T_{self.t_mod[rep]}_L_{self.t_mid[rep]}_A_{self.ancestral_size[rep]}_B_{self.mid_size[rep]}_M_{self.modern_size[rep]}_nloc_{self.num_loci[rep]}.csv" for rep in range(len(self.t_mod))]
            #print(file_paths)
            #file_path = [f"{self.prefix}/Aggregate_SFS/aSFS_LT_{self.t_long}_CT_{self.t_card}_FT_{self.t_flor}_SFS.csv"]
            #print(file_path)

            data_dict = {'t_mod':self.t_mod, 't_mid':self.t_mid,'ancestral_size':self.ancestral_size,
                         'mid_sizef':self.mid_size, 'modern_size':self.modern_size, 'num_loci':self.num_loci, 'training_labels':self.training_labels, 'path':file_paths,
                         'local_path':local_paths}
            data_df = pd.DataFrame(data_dict)
            #data_df.to_csv('/scratch/alpine/aawe2235/Fall_2022_Diascia/DL_demo/Detectability_tests/Shared_data_repo/Benchmarking/single_bottleneck_params.csv')
            data_df.to_csv(f'{self.prefix}/Marginal_3e_params.csv')
            #data_df.to_csv('/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/single_bottleneck_params.csv')
            #file_name = f"{self.prefix}_bottle_times.npy"
            #np.save(file=file_name.replace(".npy", ""), arr=data_df) 
            return data_df

    def generate_ancestry_reps(self, t_mod, t_mid, ancestral_size,
                               mid_size, modern_size, num_loci):

        num_replicates = num_loci
        demography = msprime.Demography()
        demography.add_population(initial_size=modern_size)
        demography.add_population_parameters_change(time = t_mod, population=0, initial_size=mid_size)
        demography.add_population_parameters_change(time = t_mod+t_mid, population = 0, initial_size=ancestral_size)
        #demography.add_instantaneous_bottleneck(time=bottle_time, strength=20000, population=0)
        ancestry_reps = msprime.sim_ancestry(
                samples=self.sample_size,
                demography=demography,
                num_replicates=num_replicates,
                sequence_length = 100)
        for ts in ancestry_reps:
            mutated_ts = msprime.sim_mutations(ts, rate=1e-8)
            yield mutated_ts




    def generate_SFS(self, params_tuple):
        t_mod, t_mid, ancestral_size, mid_size, modern_size, num_loci = params_tuple
        t_mod=t_mod[0]
        t_mid=t_mid[0]
        ancestral_size=ancestral_size[0]
        mid_size=mid_size[0]
        modern_size=modern_size[0]
        num_loci=num_loci[0]
        AFS_array =  np.zeros((num_loci, self.sample_size*2+1))
        trees= self.generate_ancestry_reps(t_mod, t_mid, ancestral_size,
                                                                         mid_size, modern_size, num_loci)


        mrca = np.zeros(num_loci)
        for replicate_index, ts in enumerate(trees):
            tree = ts.first() 
            mrca = tree.total_branch_length
            AFS_array[replicate_index,]= ts.allele_frequency_spectrum(polarised=False, span_normalise=False)
            summed_AFS = np.sum(AFS_array, axis=0)
        
        mrca_time_dict = {'MRCA': mrca, 't_mod':[t_mod]*num_loci}

        SFS_dict = {'SFS':summed_AFS}
        SFS_df = pd.DataFrame(SFS_dict)
        MRCA_df = pd.DataFrame(mrca_time_dict)
        MRCA_df['bad_bottle'] = MRCA_df['MRCA'] < MRCA_df['t_mod']
        bad_bottles = sum(MRCA_df['bad_bottle'])


        SFS_df.to_csv(f"{self.prefix}/SFS_3e/SFS_T_{t_mod}_L_{t_mid}_A_{ancestral_size}_B_{mid_size}_M_{modern_size}_nloc_{num_loci}.csv") 
        #MRCA_df.to_csv(f"{self.prefix}/MRCA_bottle_time.csv")
        #note here that the SFS function in tskit is a bit unique compared to empirical calculations. Since these are simulations we can calculate some extra stats
        #that go in the zeroth and final entries. They track stats that we can know because we know the ancestral states - so like an unfolded specific stat. Both matter
        #when we are working with SUBSETS of our samples. So since I am not I think these should always be zero? The zeroth entry tracks the total branch length over
        #all samples not in our subset. The final entry tracks either branches or alleles that are ancestral to our subset but still polymorphic in the entire set of
        #samples in our tree sequence. Will likely drop these assuming I dont end up finding some issue with the allele frequency spectrum calculation here compared to how
        #I'd calculate it empirically.
        return(bad_bottles)
    
        
     
    def generate_training_data(self, file_name=None):
        if file_name is None:
            params_df = self.generate_bottle_time()
        else:
            params_df = pd.read_csv(file_name)
        line_out = np.zeros(self.N_reps)
        #training_dict = {'t_mod':params_df['t_mod'], 't_mid':params_df['t_mid'], 'ancestral_size':params_df['ancestral_size'],
        #                 'mid_size':params_df['mid_size'], 'modern_size':params_df['modern_size'], 'num_loci':params_df['num_loci']}
        
        line_out=self.generate_SFS((params_df['t_mod'], params_df['t_mid'],
                                     params_df['ancestral_size'], params_df['mid_size'], params_df['modern_size'], params_df['num_loci']))

        # if __name__ =='__main__':
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
        #         line_out = executor.map(self.generate_SFS, zip(training_dict['t_bottle'], training_dict['bottle_length'],
        #                             training_dict['ancestral_size'], training_dict['bottle_size'], training_dict['modern_size']))
        # line_df = pd.DataFrame(line_out)
        # line_df.to_csv("Bad_bottle_count.csv")
        return(line_out)
    



#path = '/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/'
path = os.getcwd()
test_pop = PopGenTraining_h(
    seed=600,
    N_reps=20000,
    sample_size=8,
    training_fractions=[0.8, 0.1, 0.1],
    n_cpu=1,
    prefix=path,
)

#test_pop.generate_bottle_time()

#test_pop.generate_training_data('single_bottleneck_params.csv')
#from datetime import datetime
#startTime = datetime.now()

test_pop.generate_training_data(pars_file)

#df = pd.DataFrame()
#df[1]= bad_bots
#df.to_csv(outfile, mode='a', header=True)

#bad_bots.to_frame().to_csv(outfile)

#print(datetime.now() - startTime)


