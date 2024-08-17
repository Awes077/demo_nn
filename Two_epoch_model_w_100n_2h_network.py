
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
    print("Usage ", sys.argv[0]," <p> ")
    sys.exit()
else:
    p = int(sys.argv[1])


def get_idle_gpu():
    """
    Utility function which uses the gpustat module to select the
    least busy GPU that is available and then sets the
    CUDA_VISIBLE_DEVICES environment variable so that
    only that GPU is used

    Largely unchanged from Silas Tittes - never ran this with gpu's for training (or sims) and so this basically just sets torch's device as the cpu.
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
class TwoEpochModel():
    '''
    
    :param int seed: random seed
    :param int N_reps: number of replicates to simulate
    :param int sample_size: sample size of each simulation
    :param int t_change: time of change from modern population size to ancestral population size
    :param int modern_size: modern effective population size
    :param int ancestral_size: ancestral effective population size
    :param int n_loci: number of independent loci to simulate - as of now, all loci are of length 100
    :param float theta: theta
    :param str prefix: prefix for the output files -- full path is encouraged
    :param List[float] training_fractions: fractions of the data to use for training, validation, and testing'''


    seed: int
    N_reps: int
    sample_size: int
    num_loci: int
    training_fractions: list[float]
    prefix: str
    n_cpu: int = 1

    
    #post init we:
    #   1) check training fractions fractions sum to 1
    #   2) separate out our training, validation, and testing data
    #   3) create the labels themselves and check that there are as many labels as there are replicates to be simulated
    #   4) generate a seed
    #   5) generate parameter array of zeroes for each parameter for our 2 epoch model.

    def __post_init__(self):
        #check that our test/train/validation split sums to 1
        if sum(self.training_fractions) != 1:
            raise ValueError("training_fractions must sum to 1")
        training_fraction, validation_fraction, test_fraction = self.training_fractions
        # list of labels for each simulation
        self.training_labels = ["training"] * int(training_fraction * self.N_reps) + \
                               ["validation"] * int(validation_fraction * self.N_reps) + \
                               ["test"] * int(test_fraction * self.N_reps)


        if len(self.training_labels) != self.N_reps:
            raise ValueError("wrong number of training labels. Check training_fractions")
        # seed the random number generator
        self.rng = default_rng(self.seed)
        self.ancestral_size = np.zeros(self.N_reps)
        self.bottle_size = np.zeros(self.N_reps)
        self.modern_size = np.zeros(self.N_reps)
        self.t_bottle = np.zeros(self.N_reps)
        self.bottle_length = np.zeros(self.N_reps)


    #generates the actual parameters for simulations from msprime, here rounded to three decimals.
    #Priors are:
    #   1) Ancestral pop size: 500:500,000, uniform
    #   3) modern pop size: 500:500,000, uniform
    #   4) time of change: 100:4*modern pop size, uniform - note that the upper limit here is the expectation for coalescence time of the
    #       current pop moving backwards in time. The goal there is to avoid situations where we draw parameters for the time of demographic
    #       change that fall after the sample has fully coalesced, at which point the change would have no effect.

    def generate_bottle_time(self):
            self.ancestral_size = self.rng.uniform(500,500000, size=self.N_reps).round(3)
            self.modern_size = self.rng.uniform(500, 500000, size=self.N_reps).round(3)
            self.t_change = self.rng.uniform(100,4*self.modern_size, size=self.N_reps).round(3)
            file_paths = [f"{self.prefix}/Two_epoch_data/SFS_T_{self.t_change[rep]}_A_{self.ancestral_size[rep]}_M_{self.modern_size[rep]}.csv" for rep in range(len(self.t_change))]
            local_paths = [f"/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/Benchmarking/Two_epoch_data/SFS_T_{self.t_change[rep]}_A_{self.ancestral_size[rep]}_M_{self.modern_size[rep]}.csv" for rep in range(len(self.t_change))]
            data_dict = {'t_change':self.t_change, 'ancestral_size':self.ancestral_size,
                          'modern_size':self.modern_size, 'training_labels':self.training_labels, 'path':file_paths, 'local_path':local_paths}
            data_df = pd.DataFrame(data_dict)
            data_df.to_csv(f'{self.prefix}/20k_two_epoch_params.csv')

            return data_df
    
    def generate_ancestry_reps(self, model, bottle_time, bottle_length, ancestral_size,
                               bottle_size, modern_size, num_loci):

        num_replicates = num_loci
        
        demography = msprime.Demography()
        demography.add_population(initial_size=modern_size)
        demography.add_population_parameters_change(time = bottle_time, populaton=0, initial_size=bottle_size)
        demography.add_population_parameters_change(time = bottle_time+bottle_length, population = 0, initial_size=ancestral_size)
        ancestry_reps = msprime.sim_ancestry(
                samples=self.sample_size,
                demography=demography,
                num_replicates=num_replicates,
                squence_length = 100)
        for ts in ancestry_reps:
            mutated_ts = msprime.sim_mutations(ts, rate=1e-8)
            yield mutated_ts




    def generate_SFS(self, params_tuple):
        t_bottle, bottle_length, ancestral_size, bottle_size, modern_size = params_tuple
        AFS_array =  np.zeros((self.num_loci, self.sample_size*2+1))
        trees = self.generate_ancestry_reps(t_bottle, bottle_length, ancestral_size,
                                                                         bottle_size, modern_size, self.num_loci)
        for replicate_index, ts in enumerate(trees):
            AFS_array[replicate_index,]= ts.allele_frequency_spectrum(polarised=False, span_normalise=False)
            summed_AFS = np.sum(AFS_array, axis=0)

        SFS_dict = {'SFS':summed_AFS}
        SFS_df = pd.DataFrame(SFS_dict)
        SFS_df.to_csv(f"{self.prefix}/Full_bottle_three_epoch/SFS_T_{t_bottle}_L_{bottle_length}_A_{ancestral_size}_B_{bottle_size}_M_{modern_size}.csv") 

        #note here that the SFS function in tskit is a bit unique compared to empirical calculations. Since these are simulations we can calculate some extra stats
        #that go in the zeroth and final entries. They track stats that we can know because we know the ancestral states - so like an unfolded specific stat. Both matter
        #when we are working with SUBSETS of our samples. So since I am not I think these should always be zero? The zeroth entry tracks the total branch length over
        #all samples not in our subset. The final entry tracks either branches or alleles that are ancestral to our subset but still polymorphic in the entire set of
        #samples in our tree sequence. Will likely drop these assuming I dont end up finding some issue with the allele frequency spectrum calculation here compared to how
        #I'd calculate it empirically.
        return(SFS_df)
    
        
     
    def generate_training_data(self, file_name=None):
        line_out = []
        if file_name is None:
            params_df = self.generate_bottle_time()
        else:
            params_df = pd.read_csv(file_name)
        training_dict = {'t_bottle':params_df['t_bottle'], 'bottle_length':params_df['bottle_length'], 'ancestral_size':params_df['ancestral_size'],
                         'bottle_size':params_df['bottle_size'], 'modern_size':params_df['modern_size']}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            line_out = executor.map(self.generate_SFS, zip(training_dict['t_bottle'], training_dict['bottle_length'],
                                    training_dict['ancestral_size'], training_dict['bottle_size'], training_dict['modern_size']))
        
        return(params_df)


path = os.getcwd()
test_pop = TwoEpochModel(
    seed=4,
    N_reps=5000,
    sample_size=8,
    training_fractions=[0.8, 0.1, 0.1],
    n_cpu=p,
    prefix=path,
    num_loci=25000
)


big_test = pd.read_csv('/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/Benchmarking/Empirical_run/Flower_reserve_floribunda/20k_2e_FR_flor_params.csv')
#path for 
#big_test = pd.read_csv('/scratch/alpine/aawe2235/Fall_2022_Diascia/DL_demo/Detectability_tests/Shared_data_repo/Benchmarking/single_bottleneck_params.csv')

big_test['training_labels'] = np.random.permutation(big_test['training_labels'].values)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#This is what lets us join our observed data with our labels basically, so our predictor and hypothesis class. This is also where I'm running into issues with 
#python pickle biz. Pickles are serialized data structures from python - basically writing python objects straight to disk. For whatever reason my csvs are getting
#loaded as pickles oh wait is it because I'm using np.load instead of pd something or other, since I'm writing the csv using pd.to_csv?? Hmm....seems to have cleared that up
#so just make sure to keep using pd rather than np lol


class PopulationDataset(Dataset):
    #upon initialization, define training labels and separate out the popgen_training element
    def __init__(self, df, popgen_training: PopGenTraining_h, training_label = "training"): 
        self.df = df.query(f"training_labels == '{training_label}'")
        self.popgen_training = popgen_training
    def __getitem__(self, index):
        # get positions and convert into a tensor in torch
        dats = pd.read_csv(self.df.iloc[index].local_path)
        SFS = torch.tensor(dats.iloc[:,1].values).to(torch.float64)
        full_dat = self.df[['t_change','ancestral_size', 'modern_size']].iloc[index]
        y_dat = torch.tensor(full_dat.values, dtype=torch.float64)
        return SFS.to(device), y_dat.to(device)
    
    #define the length as well
    def __len__(self):
       return len(self.df)
    


training_dfs = [PopulationDataset(big_test ,test_pop, training_label = t) for t in ["training", "validation", "test"]]


training_dfs[0]

training_loader, validation_loader, test_loader = [DataLoader(dataset=df, batch_size=100, shuffle=True) for df in training_dfs]



import torch.nn as nn

# #make a model dataclass out of the nn module

class MultiOutputRegression(nn.Module):
    #define elements of that, in particular input_size, hidden_size, hidden_count, and output_size
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiOutputRegression, self).__init__()

        #initializing the dimensions of the ntwork here, 1 linear input, 2 hidden linear layers, and a linear output layer.
        #note that I was a bit lazy here and just defined each layer by hand, but there are a few different ways to define larger numbers
        #of hidden layers for deeper architectures.
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.double()

    def forward(self, x):
        #defining layer1 as the input x
        x = self.layer1(x)
        #calling the nn.ReLU (Rectified Linear Unit) function. Activation function that transforms our input into the activation or output of a node. Basically returns
        #the raw input if positive and a 0 if negative I think? Used as the activation function for regression problems, as it spits out an untransformed value and keeps values
        #from being negative.
        x = nn.ReLU()(x)
        #repeat for hidden layers
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)
        x = nn.ReLU()(x)
        x = self.layer4(x)
        #no transform for output

        return x.to(torch.float64)
    
loss_fn = nn.MSELoss()
output_size = 3
aSFS_len = ((test_pop.sample_size*2)+1) #SFS length here is twice the sample size (sample size in diploid individuals) + 1 - this is the output size for the SFS from msprime. Keep
                                        #in mind that since I am working with the folded SFS, this is only going to be about half full! The back half will all be zeros.

#Define model dimensions - in particular the num size of input, hidden, and output layers.
model = MultiOutputRegression(aSFS_len, 200, output_size).to(device)
#Just using the Adam optimizer with a fairly modest learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#Defining scale parameters to scale down our labels - otherwise no learning really occurs, trust me.
scale_tens = torch.tensor([2*250250,250250,250250])

for epoch in range(30):
    for x_batch, y_batch in training_loader:
        # 1. Generate predictions
        y_batch = torch.div(y_batch, scale_tens)
        pred = model(x_batch)
        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)
        # 3. Compute gradients
        loss.backward()
        # 4. Update parameters using gradients
        #when necessary can define clipping parameters to aid in learning - just puts a cap on error derivatives.
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        # 5. Reset the gradients to zero
        optimizer.zero_grad()
    #print a lil report every 2 epochs
    if epoch % 2 == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')



torch.save(model.state_dict(), '2e_large_sample_single_n100_2h.pt')
fig_train_bot_t, (ax_train_bot_t, ax_train_anc_size,ax_train_mod_size) = plt.subplots(3,1, figsize=(15,15))
fig_train_bot_t.tight_layout()


for x_batch, y_batch in training_dfs[0]:
    y_batch = torch.div(y_batch,scale_tens)
    y_na = y_batch.numpy()
    pred = model(x_batch)
    pred_na = pred.detach().numpy()
    ax_train_bot_t.scatter(2*250250*y_na[0], 2*250250*pred_na[0], color = 'black')
    ax_train_anc_size.scatter(250250*y_na[1], 250250*pred_na[1], color = 'black')
    ax_train_mod_size.scatter(250250*y_na[2], 250250*pred_na[2], color = 'black')
time_one_one_train = np.linspace(100,1.5e6,100)
ax_train_bot_t.plot(time_one_one_train, time_one_one_train)
ax_train_bot_t.set_xlabel("Simulated bottleneck time")
ax_train_bot_t.set_ylabel("Predicted bottleneck time")
anc_one_one_train = np.linspace(100,500000,100)
ax_train_anc_size.plot(anc_one_one_train, anc_one_one_train)
ax_train_anc_size.set_xlabel("Simulated ancestral size")
ax_train_anc_size.set_ylabel("Predicted ancestral size")
ms_one_one_train = np.linspace(100,500000,100)
ax_train_mod_size.plot(ms_one_one_train, ms_one_one_train)
ax_train_mod_size.set_xlabel("Simulated modern size")
ax_train_mod_size.set_ylabel("Predicted modern size")
plt.savefig('2e_train_plot_2h.png')



fig_val_bot_t, (ax_val_bot_t, ax_val_anc_size,ax_val_mod_size) = plt.subplots(3,1, figsize=(15,15))
fig_val_bot_t.tight_layout()

for x_batch, y_batch in training_dfs[1]:
    y_batch = torch.div(y_batch,scale_tens)
    y_na = y_batch.numpy()
    pred = model(x_batch)
    pred_na = pred.detach().numpy()
    ax_val_bot_t.scatter(2*250250*y_na[0], 2*250250*pred_na[0], color = 'black')
    #ax_val_bott_len.scatter(50000*y_na[1], 50000*pred_na[1], color = 'black')
    ax_val_anc_size.scatter(250250*y_na[1], 250250*pred_na[1], color = 'black')
    #ax_val_bott_size.scatter(250250*y_na[3], 250250*pred_na[3], color = 'black')
    ax_val_mod_size.scatter(250250*y_na[2], 250250*pred_na[2], color = 'black')
time_one_one_val = np.linspace(100,1.5e6,100)
ax_val_bot_t.plot(time_one_one_val, time_one_one_val)
ax_val_bot_t.set_xlabel("Simulated bottleneck time")
ax_val_bot_t.set_ylabel("Predicted bottleneck time")
anc_one_one_val = np.linspace(100,500000,100)
ax_val_anc_size.plot(anc_one_one_val, anc_one_one_val)
ax_val_anc_size.set_xlabel("Simulated ancestral size")
ax_val_anc_size.set_ylabel("Predicted ancestral size")
ms_one_one_val = np.linspace(100,500000,100)
ax_val_mod_size.plot(ms_one_one_val, ms_one_one_val)
ax_val_mod_size.set_xlabel("Simulated modern size")
ax_val_mod_size.set_ylabel("Predicted modern size")
plt.savefig('2e_val_plot_2h.png')

fig_test_bot_t, (ax_test_bot_t,  ax_test_anc_size,ax_test_mod_size) = plt.subplots(3,1,figsize=(15,15))
fig_test_bot_t.tight_layout()

for x_batch, y_batch in training_dfs[2]:
    y_batch = torch.div(y_batch,scale_tens)
    y_na = y_batch.numpy()
    pred = model(x_batch)
    pred_na = pred.detach().numpy()
    ax_test_bot_t.scatter(2*250250*y_na[0], 2*250250*pred_na[0], color = 'black')
    ax_test_anc_size.scatter(250250*y_na[1], 250250*pred_na[1], color = 'black')
    ax_test_mod_size.scatter(250250*y_na[2], 250250*pred_na[2], color = 'black')
time_one_one_test = np.linspace(100,1.5e6,100)
ax_test_bot_t.plot(time_one_one_test, time_one_one_test)
ax_test_bot_t.set_xlabel("Simulated bottleneck time")
ax_test_bot_t.set_ylabel("Predicted bottleneck time")
anc_one_one_test = np.linspace(100,500000,100)
ax_test_anc_size.plot(anc_one_one_test, anc_one_one_test)
ax_test_anc_size.set_xlabel("Simulated ancestral size")
ax_test_anc_size.set_ylabel("Predicted ancestral size")
ms_one_one_test = np.linspace(100,500000,100)
ax_test_mod_size.plot(ms_one_one_test, ms_one_one_test)
ax_test_mod_size.set_xlabel("Simulated modern size")
ax_test_mod_size.set_ylabel("Predicted modern size")
plt.savefig('2e_test_plot_2h.png')

for x_batch, y_batch in validation_loader:
    y_batch = torch.div(y_batch, scale_tens)
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    print(f'Loss is {loss.item():.4f}')

for x_batch, y_batch in test_loader:
    y_batch = torch.div(y_batch, scale_tens)
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    print(f'Loss is {loss.item():.4f}')






