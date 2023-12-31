import numpy as np
from flask import Flask, request, render_template
import pickle
app = Flask(__name__,template_folder = '../Zpmato Rest')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictor',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))
