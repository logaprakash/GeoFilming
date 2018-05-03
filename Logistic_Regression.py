"""
Logistic Regresion
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
#import numpy as np

dataset_df = pd.read_pickle('mode_binning_with_states.p')
data = dataset_df.drop(columns=["Rating","all"])
target = dataset_df["Rating"]
X_train, X_test, Y_train, Y_test = data[:800000], data[800000:], target[:800000], target[800000:]
logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
logReg_predictions = logReg.predict(X_test)
print("Accuracy:"+ str(logReg.score(X_test,Y_test)))
    
#    scores = cross_val_score(logReg, data, target, cv=5)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

logReg1 = LogisticRegression()
logReg1.fit(data, target)

filename = 'logistic_model_with_state.p'

pickle.dump(logReg1, open(filename, 'wb'))
print("files created")
    
sample_input = data.loc[[1]]
sample_input.replace(1,0)
sample_input.to_pickle("sample_input.p")
#
#
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.predict(a)