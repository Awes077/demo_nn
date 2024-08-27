
import numpy as np
from numpy.random import default_rng
import pandas as pd
import msprime
import tskit
import torch
import gpustat
import os
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
from sklearn.multioutput import MultiOutputRegressor#, RegressorChain
from sklearn.model_selection import train_test_split#, learning_curve, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import matplotlib.pyplot as plt
import os
import zipfile

from dataclasses import dataclass


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
    :param int n_loci: number of independent loci to simulate 
    :param int loc_len: length of each locus simulated, sticking to short reads here, so between 80 and 250 bp

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
        self.loc_len = np.zeros(self.N_reps)



    def generate_bottle_time(self):
            self.ancestral_size = self.rng.uniform(500,500000, size=self.N_reps).round(3)
            self.mid_size = self.rng.uniform(500, 500000, size=self.N_reps).round(3)
            self.modern_size = self.rng.uniform(500, 500000, size=self.N_reps).round(3)
            self.t_mod = self.rng.uniform(100,4*self.modern_size, size=self.N_reps).round(3)
            self.t_mid = self.rng.uniform(0, 4*self.mid_size, size=self.N_reps).round(3)
            self.num_loci = self.rng.uniform(20000, 100000, size=self.N_reps).astype(int)
            self.loc_len = self.rng.uniform(85, 250, size = self.N_reps).astype(int)
            file_paths = [f"{self.prefix}/3e_workflow/3e_SFS_sims/SFS_T_{self.t_mod[rep]}_L_{self.t_mid[rep]}_A_{self.ancestral_size[rep]}_B_{self.mid_size[rep]}_M_{self.modern_size[rep]}_nloc_{self.num_loci[rep]}.csv" for rep in range(len(self.t_mod))]
            local_paths = [f"/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/Clean_model_scripts/3e_workflow/3e_SFS_sims/SFS_T_{self.t_mod[rep]}_L_{self.t_mid[rep]}_A_{self.ancestral_size[rep]}_B_{self.mid_size[rep]}_M_{self.modern_size[rep]}_nloc_{self.num_loci[rep]}_len_{self.loc_len}.csv" for rep in range(len(self.t_mod))]
            data_dict = {'t_mod':self.t_mod, 't_mid':self.t_mid,'ancestral_size':self.ancestral_size,
                         'mid_size':self.mid_size, 'modern_size':self.modern_size, 'num_loci':self.num_loci, 'loc_len':self.loc_len,
                         'training_labels':self.training_labels, 'path':file_paths,
                         'local_path':local_paths}
            data_df = pd.DataFrame(data_dict)
            #data_df.to_csv('/scratch/alpine/aawe2235/Fall_2022_Diascia/DL_demo/Detectability_tests/Shared_data_repo/Benchmarking/single_bottleneck_params.csv')
            data_df.to_csv(f'{self.prefix}/Marginal_3e_params.csv')
            #data_df.to_csv('/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/single_bottleneck_params.csv')
            #file_name = f"{self.prefix}_bottle_times.npy"
            #np.save(file=file_name.replace(".npy", ""), arr=data_df) 
            return data_df

    def generate_ancestry_reps(self, t_mod, t_mid, ancestral_size,
                               mid_size, modern_size, num_loci, loc_len):

        num_replicates = num_loci
        ll = loc_len
        demography = msprime.Demography()
        demography.add_population(initial_size=modern_size)
        demography.add_population_parameters_change(time = t_mod, population=0, initial_size=mid_size)
        demography.add_population_parameters_change(time = t_mod+t_mid, population = 0, initial_size=ancestral_size)
        #demography.add_instantaneous_bottleneck(time=bottle_time, strength=20000, population=0)
        ancestry_reps = msprime.sim_ancestry(
                samples=self.sample_size,
                demography=demography,
                num_replicates=num_replicates,
                sequence_length = ll)
        for ts in ancestry_reps:
            mutated_ts = msprime.sim_mutations(ts, rate=1e-8)
            yield mutated_ts




    def generate_SFS(self, params_tuple):
        t_mod, t_mid, ancestral_size, mid_size, modern_size, num_loci, loc_len = params_tuple
        t_mod=t_mod[0]
        t_mid=t_mid[0]
        ancestral_size=ancestral_size[0]
        mid_size=mid_size[0]
        modern_size=modern_size[0]
        num_loci=num_loci[0]
        loc_len = loc_len[0]
        AFS_array =  np.zeros((num_loci, self.sample_size*2+1))
        trees= self.generate_ancestry_reps(t_mod, t_mid, ancestral_size,
                                                                         mid_size, modern_size, num_loci, loc_len)


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


        SFS_df.to_csv(f"{self.prefix}/3e_workflow/3e_SFS_sims/SFS_T_{t_mod}_L_{t_mid}_A_{ancestral_size}_B_{mid_size}_M_{modern_size}_nloc_{num_loci}_loc_len_{loc_len}.csv") 
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
        line_out=self.generate_SFS((params_df['t_mod'], params_df['t_mid'],
                                     params_df['ancestral_size'], params_df['mid_size'], params_df['modern_size'], params_df['num_loci'],
                                     params_df['loc_len']))
        return(line_out)
    
    def listdir_nohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    def fit_boosted_reg_tree(self, sim_directory):
        if('.zip' in sim_directory):
            nd = sim_directory.split('.')[0]
            with zipfile.ZipFile(sim_directory) as zipr:
                sd = os.getcwd()
                zipr.extractall(sd)
            files = self.listdir_nohidden(nd)
            sim_directory = nd +'/'
        else:
            files = self.listdir_nohidden(sim_directory)
        print(sim_directory)
        fix_length = [file.replace('len', 'len_') for file in files]
        y_df = pd.DataFrame(columns=['mod_mid_change','mid_length','ancestral_size','mid_size','modern_size','num_loci','loci_len'])
        all_splits = [file.split('_') for file in fix_length]
        ind_dict = {'mod_mid_change':2,'mid_length':4,'ancestral_size':6,'mid_size':8,'modern_size':10,'num_loci':12,'loci_len':15}
        for keys,values in ind_dict.items():
            l = [line[values] for line in all_splits]
            if values == 15:
                l = [el.split('.')[0] for el in l]
            y_df[keys] = l
            y_df[keys] = y_df[keys].astype(float)
        files2 = []
        files = self.listdir_nohidden(sim_directory)
        files2 = (sim_directory + file for file in files)
        x_dat = []
        for f in files2: x_dat.append(pd.read_csv(f)['SFS'])    
        x_df = pd.DataFrame(x_dat)
        x_df[40] = y_df['num_loci'].values
        x_df[39] = y_df['loci_len'].values
        x_df[40] = x_df[40]/60000
        x_df[39] = x_df[39]/250
        y_dat = y_df[['mod_mid_change','mid_length','ancestral_size','mid_size','modern_size']]
        scale_vals = [2*250250,2*250250,250250,250250, 250250]
        scaled_y = y_dat/scale_vals
        x3_train, x3_test, y3_train, y3_test = train_test_split(x_df, scaled_y)
        best_parms = {'learning_rate':0.05,
              'max_depth': None,
              'max_features':None,
              'max_leaf_nodes': 10,
              'min_samples_split':10,
              'n_estimators':400}
        bgb = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**best_parms))
        bgb.fit(x3_train, y3_train)
        return [bgb, x3_train, y3_train, x3_test, y3_test]
    
    def plot_3e_results(self, fitted_model):
        bgb, x3_train, y3_train, x3_test, y3_test = fitted_model
        bgb_test_preds = bgb.predict(x3_test)
        bgb_test_df = pd.DataFrame(bgb_test_preds)
        bgb_train_preds = bgb.predict(x3_train)
        bgb_train_df = pd.DataFrame(bgb_train_preds)

        fig, axs = plt.subplots(2,5, figsize=(10,7))
        fig.suptitle('Boosted Regression Outputs')
        fig.tight_layout()
        print('Training MSE t mod, t mid, ancestral size, mid size, modern size:', mean_squared_error(y3_train, bgb_train_preds, multioutput='raw_values'))

        print( 'Test MSE t mod, t mid, ancestral size, mid size, modern size:', mean_squared_error(y3_test, bgb_test_preds, multioutput='raw_values'))
        axs[0,0].scatter(
            y3_test['ancestral_size']*250250,
            bgb_test_df[2]*250250,
            c=y3_test['modern_size'],
            alpha=0.5
        )
        axs[0,0].set_title('Ancestral Population Size (Test)',size = 8)
        axs[0,1].scatter(
            y3_test['modern_size']*250250,
            bgb_test_df[4]*250250,
            c=y3_test['modern_size'],
            alpha=0.5
        )
        axs[0,1].set_title('Modern Population Size (Test)',size = 8)
        axs[0,2].scatter(
            y3_test['mod_mid_change']*(2*250250),
            bgb_test_df[0]*(2*250250),
            c=y3_test['modern_size'],
            alpha=0.5
        )
        axs[0,2].set_title('Time of Modern Change (Test)',size = 8)        

        axs[0,3].scatter(
            y3_test['mid_length']*(2*250250),
            bgb_test_df[1]*(2*250250),
            c=y3_test['modern_size'],
            alpha=0.5
        )
        axs[0,3].set_title('Time of Mid Change (Test)',size = 8)  

        axs[0,4].scatter(
            y3_test['mid_size']*250250,
            bgb_test_df[3]*250250,
            c=y3_test['modern_size'],
            alpha=0.5
        )       
        axs[0,4].set_title('Mid Population Size (Test)',size = 8)  


        axs[1,0].scatter(
            y3_train['ancestral_size']*250250,
            bgb_train_df[2]*250250,
            c=y3_train['modern_size'],
            alpha=0.5
        )
        axs[1,0].set_title('Ancestral Population Size (Train)',size = 8)
        axs[1,1].scatter(
            y3_train['modern_size']*250250,
            bgb_train_df[4]*250250,
            c=y3_train['modern_size'],
            alpha=0.5  
        )
        axs[1,1].set_title('Modern Population Size (Train)',size = 8)
        axs[1,2].scatter(
            y3_train['mod_mid_change']*(2*250250),
            bgb_train_df[0]*(2*250250),
            c=y3_train['modern_size'],
            alpha=0.5
        )
        axs[1,2].set_title('Time of Modern Change (Train)',size = 8)        

        axs[1,3].scatter(
            y3_train['mid_length']*(2*250250),
            bgb_train_df[1]*(2*250250),
            c=y3_train['modern_size'],
            alpha=0.5
        )
        axs[1,3].set_title('Time of Mid Change (Train)',size = 8)  

        axs[1,4].scatter(
            y3_train['mid_size']*250250,
            bgb_train_df[3]*250250,
            c=y3_train['modern_size'],
            alpha=0.5
        )
        axs[1,4].set_title('Mid Population Size (Train)',size = 8) 




