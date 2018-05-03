# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:09:31 2018

@author: loga
"""

from flask import Flask,jsonify
from flask import request 
import pandas as pd
#from Logictic_Regression import RunModel
#import json

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict(): 
    input_data = request.get_json(force=True)
    key_list = input_data['key']
    value_list = input_data['value']
    pair = dict(zip(key_list, value_list))
    
    for key, value in pair.items():
        loaded_input[key] = value_list[value]
    
    result_linear = loaded_model_linear.predict(loaded_input)
    result_logistic = loaded_model_linear.predict(loaded_input)
    result = jsonify({'result_linear':str(result_linear[0]),'result_logistic':str(result_logistic[0])})
    return result
    
if __name__ ==  '__main__': 
    linear_model_filename = 'linear_model_with_state.p'
    logistic_model_filename = 'logistic_model_with_state.p'
    sample_input_filename = 'sample_input.p'
    loaded_model_linear = pd.read_pickle(linear_model_filename)
    loaded_mode_logistic = pd.read_pickle(logistic_model_filename)
    loaded_input = pd.read_pickle(sample_input_filename)
    app.run(host = '127.0.0.1', port = 5000,debug = True)