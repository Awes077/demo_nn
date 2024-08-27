import pandas as pd 
import sys
import os

params_file = sys.argv[1]
wd = os.getcwd()
data = pd.read_csv(params_file, index_col=0)


files = data['local_path'].str.split('SFS_3e/', expand=True)[1]

just_names = files.str.split('.csv', expand=True)[0]

files = "params_3e/params_"+files

rows = len(files)
for i in range(rows):
    data.iloc[i].to_frame().T.to_csv(files[i])


