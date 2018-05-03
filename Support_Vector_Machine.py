# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:19:11 2018

@author: loga
"""

from sklearn import svm
import pandas as pd
import pickle
import numpy as np

dataset_df = pd.read_pickle('mode_binning_with_states.p')
data = dataset_df.drop(columns=["Rating","all"])
target = dataset_df["Rating"]
X_train, X_test, Y_train, Y_test = data[:800000], data[800000:], target[:800000], target[800000:]
supportVectorMachine = svm.SVC()
supportVectorMachine.fit(X_train, Y_train)  

#svm_predictions = supportVectorMachine.predict(X_test)

scores = supportVectorMachine.score(X_test,Y_test);
print(scores)
filename = 'C:\\xamp\\htdocs\\GeoFilming\\svm_model_with_state.p'
pickle.dump(supportVectorMachine, open(filename, 'wb'))
print("files created")