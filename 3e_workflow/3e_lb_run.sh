#!/bin/bash

#SBATCH --nodes=3
#SBATCH --time 24:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=192
#SBATCH --job-name=3emarginal_run
#SBATCH --output marginal_3_epoch.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=awes077@gmail.com

module purge

# Load the Load Balancer module *first*
module load loadbalance/0.2

module load anaconda
conda activate popgenDL_crashcourse

 
$CURC_LB_BIN/mpirun lb 3e_lb_cmd_marginal 
