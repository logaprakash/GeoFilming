# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:51:56 2018

@author: loga
"""

"""
Eliminating outliers using mode binning
"""

from collections import defaultdict
from numpy import mean
import pandas as pd
dataset_df = pd.read_pickle('dataset_df_states_encoded.p')
mean_dict = defaultdict(list)

dataset_df["all"] = dataset_df[dataset_df.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(int).astype(str)),axis=1)        
        

for index, x in dataset_df.iterrows():
    mean_dict[x["all"]].append(x["Rating"])

for key, value in mean_dict.items():
    mean_dict[key] = mean(value)

some =[]
for index, x in dataset_df.iterrows():
     if not mean_dict[x["all"]] is None:
         some.append(mean_dict[x["all"]])
     else:
         some.append(x["Rating"])
         
del dataset_df["Rating"]    
dataset_df["Rating"] = some   
dataset_df.to_pickle('mean_binning_with_states.p')




#dataset_df.to_pickle('dataset_df.p')