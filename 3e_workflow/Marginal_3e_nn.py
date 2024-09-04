from dataclasses import dataclass
from typing import List
from e3_data_class import *


if len(sys.argv) != 2:
    print("Usage ", sys.argv[0]," <p> ")
    sys.exit()
else:
    p = int(sys.argv[1])

print(p)


#this device variable will get used later during training
device = torch.device(f"cuda:{get_idle_gpu()}" if torch.cuda.is_available() else "cpu")




#path = '/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/'
path = os.getcwd()
test_pop = PopGenTraining_h(
    seed=420,
    N_reps=5000,
    sample_size=20,
    training_fractions=[0.8, 0.1, 0.1],
    n_cpu=p,
    prefix=path
)

#test_pop.generate_training_data('single_bottleneck_params.csv')

#test_pop.generate_training_data('shared_bottleneck_times_5.csv')

big_test = pd.read_csv('/Users/aaronw/Desktop/Dissertation/Chapter_5_Diascia/DL_popgen/Detectability_tests/Shared_data_repo/Benchmarking/Empirical_run/Marginal_locus_run/3e_n20_marginal_params.csv')

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
        dats = pd.read_csv(self.df.iloc[index].n20_path)
        num_loci = np.divide(self.df['num_loci'].iloc[index],60000)
        #dats['SFS'].iloc[40] = num_loci
        SFS = torch.tensor(dats.iloc[:,1].values).to(torch.float64)
        SFS = SFS[:22]
        SFS[-1] = num_loci
        #print(SFS)
        #print(SFS.shape)
        #SFM = SFS!=0
        #SFS_mask = SFM.nonzero()
        #SFS = torch.reshape(SFS, (-1,))
        #print(SFS.shape)

        #same for Ne, scaled by the Ne scaling factor
        #okay probably need to think here about the Ne and how its calculated but go on - once again just turning into a tensor within torch
        #Ne = torch.tensor(self.df.iloc[index].Ne/self.popgen_training.Ne_scaling).reshape(1)
        #t_bottles = self.df.iloc[index].t_bottle
        #anc_size = self.df.iloc[index].ancestral_size
        #bott_length = self.df.iloc[index].bottle_length
        #bott_size = self.df.iloc[index].bottle_size
        #modern_size = self.df.iloc[index].modern_size
        full_dat = self.df[['t_mod','t_mid','ancestral_size','mid_size', 'modern_size']].iloc[index]

        y_dat = torch.tensor(full_dat.values, dtype=torch.float64)
        #print(y_dat)
        #anc_size_tens = torch.tensor(anc_size, dtype = torch.float64)
        #bott_len_tens = torch.tensor(bott_length, dtype = torch.float64)
        #bott_size_tens = torch.tensor(bott_size, dtype = torch.float64)
        #mod_size_tens = torch.tensor(modern_size, dtype = torch.float64)
        #create an array of zeros it looks like, same dimensions as the number of sites
        #position_array = torch.zeros(self.popgen_training.numsites, dtype=torch.float32)
        #replace with site positions divided by sequence length? so relative positions I'm guessing??
        #position_array[:len(position)] = position[:self.popgen_training.numsites]/self.popgen_training.sequence_length
        #then return the position array to the device defined earlier, which I think is our idle GPU or the cpu?

        return SFS.to(device), y_dat.to(device)
    
    #define the length as well
    def __len__(self):
       return len(self.df)
    


training_dfs = [PopulationDataset(big_test ,test_pop, training_label = t) for t in ["training", "validation", "test"]]



training_loader, validation_loader, test_loader = [DataLoader(dataset=df, batch_size=30, shuffle=True) for df in training_dfs]



import torch.nn as nn

# #make a model dataclass out of the nn module?

class MultiOutputRegression(nn.Module):
    #define elements of that, in particular input_size, hidden_size, hidden_count, and output_size
    def __init__(self, input_size, hidden_size, hidden_count, output_size):
        #okay so super just seems to a means of delegetating some sort of call? seems like its for delegating attribute calls to ancestor data classes? yeah so
        #not entirely sure what that means but it seems like there are elements that are universal to other languages and how they use super() and some use cases where
        #it is entirely idiomatic to python. Not sure which this qualifies as. Maybe this is inheriting the init from the earlier data class??

        #Okay so I'm guessing here that its like, I am creating a new class, Model, out of the extant nn.module class. So here I can define init processes taht are just
        #straight out of nn, rather than having to call them new? Or maybe this is if I make new classes out of Model, these are automatially inherited?
        super(MultiOutputRegression, self).__init__()
        #set hidden count parameter
        #self.hidden_count = hidden_count
        #set dimensions for each of the layers, here we have three layers, an input x hidden layer, a hidden x hidden layer, and a hidden x output layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.double()
        self.hidden_count = hidden_count
       
        #now we define a forward call - this is what makes our initial predictions based on the network - the example in the book also defines a backward, which is the
        #back propagation for training, but I'm guessing maybe that's actually inherent to the class? So maybe I only need to explicitly specify a forward call to set
        # the activation function (ReLU in this case), and then backward is innate to the nn.Module and just knows the gradient calculations for whatever activation
        # I set?
        

    def forward(self, x):
        #defining layer1 as the input x
        x = self.layer1(x)
        #calling the nn.ReLU (Rectified Linear Unit) function. Activation function that transforms our input into the activation or output of a node. Basically returns
        #the raw input if positive and a 0 if negative I think? Used as the activation function for regression problems, as it spits out an untransformed value? So yeah
        #for now can just think of it as a linear activation function rather than digging too deep into the assorted issues with other functions, e.g. logistic, tan, etc.
        x = nn.ELU()(x)
        # for i in 0 to hidden count, set layer 2 equal to x, then call ReLU on x. Okay so like...thing I'm not so sure about here is that I want to have more than one hidden 
        # layer. It seem slike here we loop through the count of hidden layers and call layer 2 each time? should there be some way to index the layers themselves? or set up
        # a module list like in the book? Hmm....
        for i in range(self.hidden_count):
            x = self.dropout(x)
            x = self.layer2(x)
            x = nn.ELU()(x)
        x = self.layer3(x)
        x = nn.ELU()(x)






       #for i in range(self.hidden_count):
       #     x = self.layer2(x)
       #     x = nn.ReLU()(x)
        #set layer 3 equal to x
       # x = self.layer3(x)
        #return x
        return x.to(torch.float64)
    
loss_fn = nn.MSELoss()
#loss_full = nn.MSELoss(reduction='none')
output_size = 5
#okay need to figure out how to reformat this part, in particular the element here about inut size isn't going to be numsites for me, it's going to be the length of the SFSx3
#yeah? Something like that? I guess if I just turn it into a vector it should be able to manage but I'm curious what it looks like once I've read it in.

aSFS_len = ((test_pop.sample_size)+2)

model = MultiOutputRegression(aSFS_len, 40, 2, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scale_tens = torch.tensor([2*250250,2*250250,250250,250250, 250250])

clip_value = 25
#20 training epochs
R2s = R2Score(num_outputs=5,multioutput="raw_values")

for epoch in range(20):
    for x_batch, y_batch in training_loader:
        # 1. Generate predictions
        y_batch = torch.div(y_batch, scale_tens)
        pred = model(x_batch)
        #print(pred)
        #y_batch = y_batch.unsqueeze(1)
        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)
        #lossfull = loss_full(pred,y_batch)
        sq_ten = (pred - y_batch)**2
        lossful = torch.mean(sq_ten, axis=0)


        # 3. Compute gradients
        loss.backward()
        # 4. Update parameters using gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        # 5. Reset the gradients to zero
        optimizer.zero_grad()
            

    #print a lil report every 2 epochs
    if epoch % 2 == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')
        print(f'Epoch {epoch} Loss {lossful}')
        print(f'Current R2 vector is: {R2s(y_batch, pred)}')




torch.save(model.state_dict(), 'Marginal_scaled_3e_20Samp_n100_1h.pt')
#fig_train_bot_t, (ax_train_bot_t, ax_train_bott_len, ax_train_anc_size,ax_train_bott_size,ax_train_mod_size) = plt.subplots(5,1, figsize=(15,15))
#fig_train_bot_t.tight_layout()
# fig_train_bott_len, ax_train_bott_len = plt.subplots()
# fig_train_anc_size, ax_train_anc_size = plt.subplots()
# fig_train_bott_size, ax_train_bott_size = plt.subplots()
# fig_train_mod_size, ax_train_mod_size = plt.subplots()

train_R2_scaled = pd.DataFrame(columns=range(5))
train_loss = pd.DataFrame(columns=range(1))
y_df = pd.DataFrame(columns=range(5))
pred_df = pd.DataFrame(columns=range(5))
for x_batch, y_batch in training_dfs[0]:
    y_batch = torch.div(y_batch,scale_tens)
    y_na = y_batch.numpy()
    #print(y_na)
    pred = model(x_batch)
    #print(pred.na)
    pred_na = pred.detach().numpy()
    y_df = pd.concat([y_df,pd.DataFrame([y_na])],axis=0, ignore_index=True)
    pred_df= pd.concat([pred_df,pd.DataFrame([pred_na])], ignore_index=True)
    r2_scores=R2s(pred, y_batch)
    train_R2_scaled = pd.concat([train_R2_scaled, pd.DataFrame([r2_scores])], axis=0, ignore_index=True)
    train_loss = pd.concat([train_loss, pd.DataFrame([loss.item()])], axis=0, ignore_index=True)



    # ax_train_bot_t.scatter(2*250250*y_na[0], 2*250250*pred_na[0], color = 'black')
    # ax_train_bott_len.scatter(2*250250*y_na[1], 2*250250*pred_na[1], color = 'black')
    # ax_train_anc_size.scatter(250250*y_na[2], 250250*pred_na[2], color = 'black')
    # ax_train_bott_size.scatter(250250*y_na[3], 250250*pred_na[3], color = 'black')
    # ax_train_mod_size.scatter(250250*y_na[4], 250250*pred_na[4], color = 'black')

train_R2_scaled.to_csv('3e_n20_sparse_results/1h_scaled_n20_3e_training_r_squared.csv')
train_loss.to_csv('3e_n20_sparse_results/1h_scaled_n20_3e_training_loss.csv')
y_df.to_csv('3e_n20_sparse_results/1h_scaled_n20_3e_training_y_data.csv')
pred_df.to_csv('3e_n20_sparse_results/1h_scaled_n20_3e_training_pred_data.csv')


def R_sq(preds, y_dat):
    y_bar = y_dat.mean()
    tot_var = np.square(y_dat-y_bar).sum()
    ssres = np.square(preds-y_dat).sum()
    rs = 1- np.divide(ssres,tot_var)
    return(rs)

Rs_tmod = R_sq(pred_df[0], y_df[0])
Rs_tmid = R_sq(pred_df[1], y_df[1])
Rs_na = R_sq(pred_df[2], y_df[2])
Rs_nmid = R_sq(pred_df[3], y_df[3])
Rs_nmod = R_sq(pred_df[4], y_df[4])

R_sq_df = pd.DataFrame({'Rs_tmod':Rs_tmod, 'Rs_tmid':Rs_tmid, 'Rs_na': Rs_na, 'Rs_nmid':Rs_nmid, 'Rs_nmod':Rs_nmod}, index=[0])
R_sq_df.to_csv('3e_n20_sparse_results/Training_scaled_n100_3e_1h_Rsquared_stats.csv')


plt.scatter(2*250250*y_df[0], 2*250250*pred_df[0], color='black')
time_one_one_train = np.linspace(100,1.5e6,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated modern-mid transition time")
plt.ylabel("Predicted modern-mid transition time")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_train_t_mod.png", bbox_inches="tight")
plt.close()

plt.scatter(2*250250*y_df[1], 2*250250*pred_df[1], color='black')
time_one_one_train = np.linspace(100,1.5e6,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated mid-ancestral transition time")
plt.ylabel("Predicted mid-ancestral transition time")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_train_t_mid.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[2], 250250*pred_df[2], color='black')
time_one_one_train = np.linspace(100,500000,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Ancestral Population Size")
plt.ylabel("Predicted Ancestral Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_train_anc_size.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[3], 250250*pred_df[3], color='black')
time_one_one_train = np.linspace(100,500000,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Mid-Epoch Population Size")
plt.ylabel("Predicted Mid-Epoch Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_train_mid_size.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[4], 250250*pred_df[4], color='black')
time_one_one_train = np.linspace(100,500000,100)
plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Modern Population Size")
plt.ylabel("Predicted Modern Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_train_mod_size.png", bbox_inches="tight")
plt.close()



# plt.scatter(250250*y_df[3], 250250*pred_df[3], color='black')
# time_one_one_train = np.linspace(100,500000,100)
# plt.plot(time_one_one_train, time_one_one_train)
# plt.xlabel("Simulated Middle Size")
# plt.ylabel("Predicted Middle Size")
# plt.savefig("FR_flor_3e_train_mid_size.png", bbox_inches="tight")
# plt.close()


# plt.scatter(250250*y_df[4], 250250*pred_df[4], color='black')
# time_one_one_train = np.linspace(100,500000,100)
# plt.plot(time_one_one_train, time_one_one_train)
# plt.xlabel("Simulated Modern Size")
# plt.ylabel("Predicted Modern Size")
# plt.savefig("FR_flor_3e_train_mod_size.png", bbox_inches="tight")
# plt.close()






y_df = pd.DataFrame(columns=range(5))
pred_df = pd.DataFrame(columns=range(5))
val_R2_scaled = pd.DataFrame(columns=range(5))

for x_batch, y_batch in training_dfs[1]:
    y_batch = torch.div(y_batch,scale_tens)
    y_na = y_batch.numpy()
    #print(y_na)
    pred = model(x_batch)
    #print(pred.na)
    pred_na = pred.detach().numpy()
    y_df = pd.concat([y_df,pd.DataFrame([y_na])],axis=0, ignore_index=True)
    pred_df= pd.concat([pred_df,pd.DataFrame([pred_na])], ignore_index=True)
    val_R2=R2s(pred, y_batch)
    val_R2_scaled = pd.concat([val_R2_scaled, pd.DataFrame([val_R2])], axis=0, ignore_index=True)




    # ax_train_bot_t.scatter(2*250250*y_na[0], 2*250250*pred_na[0], color = 'black')
    # ax_train_bott_len.scatter(2*250250*y_na[1], 2*250250*pred_na[1], color = 'black')
    # ax_train_anc_size.scatter(250250*y_na[2], 250250*pred_na[2], color = 'black')
    # ax_train_bott_size.scatter(250250*y_na[3], 250250*pred_na[3], color = 'black')
    # ax_train_mod_size.scatter(250250*y_na[4], 250250*pred_na[4], color = 'black')

val_R2_scaled.to_csv('1h_scaled_n20_3e_val_r_squared.csv')
y_df.to_csv('1h_scaled_n20_3e_val_y_data.csv')
pred_df.to_csv('1h_scaled_n20_3e_val_pred_data.csv')

Rs_tmod = R_sq(pred_df[0], y_df[0])
Rs_tmid = R_sq(pred_df[1], y_df[1])
Rs_na = R_sq(pred_df[2], y_df[2])
Rs_nmid = R_sq(pred_df[3], y_df[3])
Rs_nmod = R_sq(pred_df[4], y_df[4])

R_sq_df = pd.DataFrame({'Rs_tmod':Rs_tmod, 'Rs_tmid':Rs_tmid, 'Rs_na': Rs_na, 'Rs_nmid':Rs_nmid, 'Rs_nmod':Rs_nmod}, index=[0])
R_sq_df.to_csv('3e_n20_sparse_results/Validation_scaled_n100_3e_1h_Rsquared_stats.csv')

plt.scatter(2*250250*y_df[0], 2*250250*pred_df[0], color='black')
time_one_one_train = np.linspace(100,1.5e6,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated modern-mid transition time")
plt.ylabel("Predicted modern-mid transition time")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_val_t_mod.png", bbox_inches="tight")
plt.close()

plt.scatter(2*250250*y_df[1], 2*250250*pred_df[1], color='black')
time_one_one_train = np.linspace(100,1.5e6,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated mid-ancestral transition time")
plt.ylabel("Predicted mid-ancestral transition time")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_val_t_mid.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[2], 250250*pred_df[2], color='black')
time_one_one_train = np.linspace(100,500000,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Ancestral Population Size")
plt.ylabel("Predicted Ancestral Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_val_anc_size.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[3], 250250*pred_df[3], color='black')
time_one_one_train = np.linspace(100,500000,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Mid-Epoch Population Size")
plt.ylabel("Predicted Mid-Epoch Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_val_mid_size.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[4], 250250*pred_df[4], color='black')
time_one_one_train = np.linspace(100,500000,100)
plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Modern Population Size")
plt.ylabel("Predicted Modern Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_val_mod_size.png", bbox_inches="tight")
plt.close()



# plt.scatter(250250*y_df[3], 250250*pred_df[3], color='black')
# time_one_one_train = np.linspace(100,500000,100)
# plt.plot(time_one_one_train, time_one_one_train)
# plt.xlabel("Simulated Middle Size")
# plt.ylabel("Predicted Middle Size")
# plt.savefig("FR_flor_3e_val_mid_size.png", bbox_inches="tight")
# plt.close()


# plt.scatter(250250*y_df[4], 250250*pred_df[4], color='black')
# time_one_one_train = np.linspace(100,500000,100)
# plt.plot(time_one_one_train, time_one_one_train)
# plt.xlabel("Simulated Modern Size")
# plt.ylabel("Predicted Modern Size")
# plt.savefig("FR_flor_3e_val_mod_size.png", bbox_inches="tight")
# plt.close()


y_df = pd.DataFrame(columns=range(5))
pred_df = pd.DataFrame(columns=range(5))
test_R2_scaled = pd.DataFrame(columns=range(5))

for x_batch, y_batch in training_dfs[2]:
    y_batch = torch.div(y_batch,scale_tens)
    y_na = y_batch.numpy()
    #print(y_na)
    pred = model(x_batch)
    #print(pred.na)
    pred_na = pred.detach().numpy()
    y_df = pd.concat([y_df,pd.DataFrame([y_na])],axis=0, ignore_index=True)
    pred_df= pd.concat([pred_df,pd.DataFrame([pred_na])], ignore_index=True)
    test_R2=R2s(pred, y_batch)
    test_R2_scaled = pd.concat([test_R2_scaled, pd.DataFrame([test_R2])], axis=0, ignore_index=True)



    # ax_train_bot_t.scatter(2*250250*y_na[0], 2*250250*pred_na[0], color = 'black')
    # ax_train_bott_len.scatter(2*250250*y_na[1], 2*250250*pred_na[1], color = 'black')
    # ax_train_anc_size.scatter(250250*y_na[2], 250250*pred_na[2], color = 'black')
    # ax_train_bott_size.scatter(250250*y_na[3], 250250*pred_na[3], color = 'black')
    # ax_train_mod_size.scatter(250250*y_na[4], 250250*pred_na[4], color = 'black')

test_R2_scaled.to_csv('1h_scaled_n20_3e_test_r_squared.csv')
y_df.to_csv('1h_scaled_n20_3e_test_y_data.csv')
pred_df.to_csv('1h_scaled_n20_3e_test_pred_data.csv')

Rs_tmod = R_sq(pred_df[0], y_df[0])
Rs_tmid = R_sq(pred_df[1], y_df[1])
Rs_na = R_sq(pred_df[2], y_df[2])
Rs_nmid = R_sq(pred_df[3], y_df[3])
Rs_nmod = R_sq(pred_df[4], y_df[4])

R_sq_df = pd.DataFrame({'Rs_tmod':Rs_tmod, 'Rs_tmid':Rs_tmid, 'Rs_na': Rs_na, 'Rs_nmid':Rs_nmid, 'Rs_nmod':Rs_nmod}, index=[0])
R_sq_df.to_csv('3e_n20_sparse_results/Testingn_scaled_n100_3e_1h_Rsquared_stats.csv')


plt.scatter(2*250250*y_df[0], 2*250250*pred_df[0], color='black')
time_one_one_train = np.linspace(100,1.5e6,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated modern-mid transition time")
plt.ylabel("Predicted modern-mid transition time")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_test_t_mod.png", bbox_inches="tight")
plt.close()

plt.scatter(2*250250*y_df[1], 2*250250*pred_df[1], color='black')
time_one_one_train = np.linspace(100,1.5e6,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated mid-ancestral transition time")
plt.ylabel("Predicted mid-ancestral transition time")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_test_t_mid.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[2], 250250*pred_df[2], color='black')
time_one_one_train = np.linspace(100,500000,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Ancestral Population Size")
plt.ylabel("Predicted Ancestral Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_test_anc_size.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[3], 250250*pred_df[3], color='black')
time_one_one_train = np.linspace(100,500000,100)

plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Mid-Epoch Population Size")
plt.ylabel("Predicted Mid-Epoch Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_test_mid_size.png", bbox_inches="tight")
plt.close()

plt.scatter(250250*y_df[4], 250250*pred_df[4], color='black')
time_one_one_train = np.linspace(100,500000,100)
plt.plot(time_one_one_train, time_one_one_train)
plt.xlabel("Simulated Modern Population Size")
plt.ylabel("Predicted Modern Population Size")
plt.savefig("3e_n20_sparse_results/1h_scaled_marginal3e_test_mod_size.png", bbox_inches="tight")
plt.close()




# plt.scatter(250250*y_df[3], 250250*pred_df[3], color='black')
# time_one_one_train = np.linspace(100,500000,100)
# plt.plot(time_one_one_train, time_one_one_train)
# plt.xlabel("Simulated Middle Size")
# plt.ylabel("Predicted Middle Size")
# plt.savefig("FR_flor_3e_test_mid_size.png", bbox_inches="tight")
# plt.close()


# plt.scatter(250250*y_df[4], 250250*pred_df[4], color='black')
# time_one_one_train = np.linspace(100,500000,100)
# plt.plot(time_one_one_train, time_one_one_train)
# plt.xlabel("Simulated Modern Size")
# plt.ylabel("Predicted Modern Size")
# plt.savefig("FR_flor_3e_test_mod_size.png", bbox_inches="tight")
# plt.close()

# fig_train_bot_t, ax_train_bot_t = plt.subplots()
# for x_batch, y_batch in training_dfs[0]:
#     y_batch = torch.div(y_batch,250250)
#     y_na = y_batch.numpy()
#     pred = model(x_batch)
#     pred_na = pred.detach.numpy()
#     ax_train_bot_t.scatter(y_na[0], pred_na[0], color = 'black')
# one_one_train = np.linspace(100,500000,100)
# ax_train_bot_t.plot(one_one_train, one_one_train)
# ax_train_bot_t.set_xlabel("Simulated bottleneck time")
# ax_train_bot_t.set_ylabel("Predicted bottleneck time")
# plt.savefig('scaled_train_plot_t_bott.png')

# val_fig, val_ax = plt.subplots()
#  #grad predictions from model and plot against the actual values I think
# for x_batch, y_batch in training_dfs[1]:
#     y_batch = torch.div(y_batch,250250)
#     y_na = y_batch.numpy()
#     pred = model(x_batch)
#     pred_na = pred.detach.numpy()

#     val_ax.scatter(y_batch.item(), pred.item(), color = 'black')
# one_one = np.linspace(100, 10000, 100)
# val_ax.plot(one_one, one_one)
# val_ax.set_xlabel("Simulated bottleneck time")
# val_ax.set_ylabel("Predicted bottleneck time")
# plt.savefig('scaled_validation_plot.png')


# # #plot some figs
# fig, ax = plt.subplots()
#  #grad predictions from model and plot against the actual values I think
# for x_batch, y_batch in training_dfs[2]:
#     y_batch = torch.div(y_batch,250250)
#     pred = model(x_batch)

#     ax.scatter(y_batch.item(), pred.item(), color = 'black')
# one_one = np.linspace(100, 500000, 100)
# ax.plot(one_one, one_one)
# ax.set_xlabel("Simulated bottleneck time")
# ax.set_ylabel("Predicted bottleneck time")
# plt.savefig('scaled_test_plot.png')



# for x_batch, y_batch in validation_loader:
#     y_batch = torch.div(y_batch, scale_tens)
#     pred = model(x_batch)
#     #y_batch = y_batch.unsqueeze(1)

#     loss = loss_fn(pred, y_batch)

#     print(f'Loss is {loss.item():.4f}')



# for x_batch, y_batch in test_loader:
#     y_batch = torch.div(y_batch, scale_tens)
#     pred = model(x_batch)
#     #y_batch= y_batch.unsqueeze(1)
#     loss = loss_fn(pred, y_batch)
#     print(f'Loss is {loss.item():.4f}')






