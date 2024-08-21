#!/bin/bash

while read fil
do
 echo python Marginal_3e_sims.py params_3e/$fil >> 3e_lb_cmd_marginal
done < 3e_filelist.txt
