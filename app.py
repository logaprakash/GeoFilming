# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:09:31 2018

@author: loga
"""

from flask import Flask,jsonify
from flask import request 
import pandas as pd
#from Logictic_Regression import RunModel
import json

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict(): 
    loaded_input = pd.read_pickle(sample_input_filename)
    
    input_data = request.get_json(force=True)
    key_list = input_data['key']
    
   
    for key in key_list:
        loaded_input[key] =1
    
    result_linear = loaded_model_linear.predict(loaded_input)
    result_logistic = loaded_mode_logistic.predict(loaded_input)
    
    if result_linear[0]<=0:
        result_linear[0] = 1
        
    result = jsonify({'result_linear':str(abs(result_linear[0])*20),'result_logistic':str(abs(result_logistic[0])*20)})
    del loaded_input
    return result

@app.route('/predict-state',methods=['POST'])
def predict_state(): 

    loaded_input = pd.read_pickle(sample_input_filename)
    
    input_data = request.get_json(force=True)
    key_list = input_data['key']
    
    states = list(loaded_input)[20:]
    
    results = {}
    
    for key in key_list:
        loaded_input[key] =1
        
    for state in states:
        
        loaded_input[state] = 1
        result_linear = loaded_model_linear.predict(loaded_input)
        result_logistic = loaded_mode_logistic.predict(loaded_input)
        
        if result_linear[0]<0:
            result_linear[0] = 1
        
        results.setdefault(state, []).append(str(abs(result_linear[0])*20))
        results.setdefault(state, []).append(str(abs(result_logistic[0])*20))
        loaded_input[state] = 0
        
    result = json.dumps(results) 
    del loaded_input
    return result

@app.route('/predict-genre',methods=['POST'])
def predict_genre(): 

    loaded_input = pd.read_pickle(sample_input_filename)
    
    input_data = request.get_json(force=True)
    key_list = input_data['key']
    
    genres = list(loaded_input)[:20]
    
    results = {}
    
    for key in key_list:
        loaded_input[key] = 1
        
    for genre in genres:
        
        loaded_input[genre] = 1
        result_linear = loaded_model_linear.predict(loaded_input)
        result_logistic = loaded_mode_logistic.predict(loaded_input)
        
        if result_linear[0]<0:
            result_linear[0] = 1
        
        results.setdefault(genre, []).append(str(abs(result_linear[0])*20))
        results.setdefault(genre, []).append(str(abs(result_logistic[0])*20))
        loaded_input[genre] = 0
        
    result = json.dumps(results) 
    del loaded_input
    return result
    
if __name__ ==  '__main__': 
    linear_model_filename = 'linear_model_with_state.p'
    logistic_model_filename = 'logistic_model_with_state.p'
    sample_input_filename = 'sample_input.p'
    loaded_model_linear = pd.read_pickle(linear_model_filename)
    loaded_mode_logistic = pd.read_pickle(logistic_model_filename)
    
    app.run(host = '127.0.0.1', port = 5000,debug = True)