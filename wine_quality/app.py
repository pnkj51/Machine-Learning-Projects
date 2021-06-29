from flask import Flask, render_template, request, Markup
import io, os, sys
import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import pickle


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

features = ['fixed acidity','volatile acidity','citric acid','residual sugar',
			'chlorides','free sulfur dioxide','total sulfur dioxide',
			'density','pH','sulphates','alcohol','quality','color']

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/process', methods=['POST', 'GET'])
def process():
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	pred = model.predict(final_features)
	#output = np.round(pred,2)
	return render_template('index.html',predicted_test='wine quality is {}'.format(pred))

if __name__=='__main__':
	app.run(debug=True)