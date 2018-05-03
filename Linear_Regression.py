"""
Linear Regresion
"""

from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
import pandas as pd
#import numpy as np
import pickle

dataset_df = pd.read_pickle('mean_binning_with_states.p')
data = dataset_df.drop(columns=["Rating","all"])
target = dataset_df["Rating"]
X_train, X_test, Y_train, Y_test = data[:800000], data[800000:], target[:800000], target[800000:]
linReg = LinearRegression()
linReg.fit(X_train, Y_train)
linReg_prediction = linReg.predict(X_test)
print("Accuracy:"+ str(linReg.score(X_test,Y_test)));


linReg1 = LinearRegression()
linReg1.fit(data, target)

filename = 'linear_model_with_state.p'
pickle.dump(linReg1, open(filename, 'wb'))

sample_input_filename = 'sample_input.p'
loaded_input = pd.read_pickle(sample_input_filename)
loaded_input["comedy"]=1
loaded_input["in"]=1
linReg1.predict(loaded_input)
#scores = cross_val_score(linReg, data, target, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#filename = 'C:\\xamp\\htdocs\\GeoFilming\\model.p'
#pickle.dump(linReg, open(filename, 'wb'))
#
#
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.predict(a)

