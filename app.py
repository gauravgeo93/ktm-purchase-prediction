# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:06:30 2020

@author: Gaurav Verma
"""

import numpy as np
import pickle
from flask import Flask,render_template,jsonify,request

#Innitializing the Flask Application:
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

#Routing the application to the root folder:
@app.route('/')
def home():
    return render_template('index.html')

#Routing to the prediction outcome:
@app.route('/predict',methods=['POST'])    
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict_proba(final_data)
    output = prediction[0][1]
    return render_template('index.html',prediction_text='Probability of purchasing the Bike is {}%'.format(round((output*100),2)))

if __name__ == '__main__':
    app.run(debug=True)