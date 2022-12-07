# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:05:08 2022

@author: amrane
"""

from flask import Flask, render_template, request
import re 
from collections import Counter
from joblib import load
import numpy as np
import pickle
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', title = "INVEST NBA")


@app.route('/prediction', methods = ['POST'])
def page_pred():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    loaded_model = pickle.load(open( 'Random forest.sav', 'rb'))   
    prediction = loaded_model.predict(final_features)[0]
    
    if prediction == 1:
        return render_template('invest.html', title = "True Spam")
    else:
        return render_template('not_invest.html', title = "False Spam")
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)

