from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('marks_predictor.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    f1 = request.form['study_hours']
    pred = model.predict([[f1]])
    output = np.round(pred,2)
    return render_template('index.html',prediction_text='your marks will be ${}'.format(output))


if __name__ == '__main__':
    app.run(debug=False,port=5002)
