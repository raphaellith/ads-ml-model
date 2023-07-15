import pandas as pd
import numpy as np



from flask import Flask, render_template, url_for, request
from flask import jsonify

import pickle


app = Flask(__name__)
model = pickle.load(open('ads_pred_model.pkl', 'rb'))

@app.route("/")
def index():
    return 'hello'


@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/form")
def form():
    return render_template('form.html')


@app.route("/predict", methods=['POST'])
def predict():
    time = float(request.form.get('feature1'))
    age = float(request.form.get('feature2'))
    income = float(request.form.get('feature3'))
    internet = float(request.form.get('feature4'))
    male = int(request.form.get('feature5'))

    values = [time, age, income, internet, male]
    features = np.array(values).reshape(1, -1)
    predictions = model.predict(features)

    print('done')
    print(predictions[0])

    return render_template('form.html', prediction_text=f'The prediction is {predictions[0]}')



    return 'done'


if __name__ == '__main__':
    app.run(debug=True)
