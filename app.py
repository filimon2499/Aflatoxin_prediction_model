# -*- coding: utf-8 -*-

import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
model1 = pickle.load(open('models/model_1.pkl', 'rb')) 
model2 = pickle.load(open('models/model_2.pkl', 'rb')) 

# Create application
app = Flask(__name__, static_url_path='/static')


# Bind home function to URL
@app.route('/')
def home():
    return render_template('Aflatoxin Classifier.html')

# Bind predict function to URL
@app.route('/predict_1', methods =['POST'])
def predict_1():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction1 = model1.predict(array_features)
    
    output1 = prediction1
    
    # Check the output values and retrive the result with html tag based on the value
    if output1 == 0:
        return render_template('Aflatoxin Classifier.html', 
                               result1 = 'The aflatoxin content is likely to be below!')
    else:
        return render_template('Aflatoxin Classifier.html', 
                               result1 = 'The aflatoxin content is likely to be above!')

# Bind predict function to URL
@app.route('/predict_2', methods =['POST'])
def predict_2():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction2 = model2.predict(array_features)
    
    output2 = prediction2
    
    # Check the output values and retrive the result with html tag based on the value
    if output2 == 0:
        return render_template('Aflatoxin Classifier.html', 
                               result2 = 'The aflatoxin content is likely to be below!')
    else:
        return render_template('Aflatoxin Classifier.html', 
                               result2 = 'The aflatoxin content is likely to be above!')
if __name__ == '__main__':
#Run the application
    app.run()
