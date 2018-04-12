"""
Logistic Regresion
"""
"""
Logistic Regression
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

dataset_df = pd.read_pickle('dataset_df.p')
data = dataset_df.drop(columns=["Rating"])
target = dataset_df["Rating"]
X_train, X_test, Y_train, Y_test = data[:800000], data[800000:], target[:800000], target[800000:]
logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
logReg_predictions = logReg.predict(X_test)

scores = cross_val_score(logReg, data, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
