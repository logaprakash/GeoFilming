# -*- coding: utf-8 -*-
"""
Eliminating outliers using mode binning
"""

from collections import defaultdict
from scipy.stats import mode
import pandas as pd
dataset_df = pd.read_pickle('dataset_df_states_encoded.p')
mode_dict = defaultdict(list)

        
dataset_df["all"] = dataset_df[dataset_df.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(int).astype(str)),axis=1)        
        

for index, x in dataset_df.iterrows():
    mode_dict[x["all"]].append(x["Rating"])

for key, value in mode_dict.items():
    mode_dict[key] = mode(value)[0][0]

some =[]
for index, x in dataset_df.iterrows():
     if not mode_dict[x["all"]] is None:
         some.append(mode_dict[x["all"]])
     else:
         some.append(x["Rating"])
         
del dataset_df["Rating"]    
dataset_df["Rating"] = some   
dataset_df.to_pickle('mode_binning_with_states.p')




#dataset_df.to_pickle('dataset_df.p')