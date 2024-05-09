from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('fetal_health1.pkl', 'rb'))
model1 = pickle.load(open('scale.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def f():
    return render_template("index.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Handle form submission and redirect to the prediction route
        return redirect(url_for('predict'))
    else:
        # Render the inspect.html template for form input
        return render_template('inspect.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        baseline_value = float(request.form['baseline_value'])
        accelerations = float(request.form['accelerations'])
        fetal_movement= float(request.form['fetal_movement'])
        uterine_contractions= float(request.form['uterine_contractions'])
        light_decelerations = float(request.form['light_decelerations'])
        severe_decelerations = float(request.form['severe_decelerations'])
        prolongued_decelerations = float(request.form['prolongued_decelerations'])
        abnormal_short_term_variability = float(request.form['abnormal_short_term_variability'])
        mean_value_of_short_term_variability = float(request.form['mean_value_of_short_term_variability'])
        percentage_of_time_with_abnormal_long_term_variability = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
        mean_value_of_long_term_variability = float(request.form['mean_value_of_long_term_variability'])
        histogram_width = float(request.form['histogram_width'])
        histogram_min = float(request.form['histogram_min'])
        histogram_max = float(request.form['histogram_max'])
        histogram_number_of_peaks = float(request.form['histogram_number_of_peaks'])
        histogram_number_of_zeroes = float(request.form['histogram_number_of_zeroes'])
        histogram_mode = float(request.form['histogram_mode'])
        histogram_median = float(request.form['histogram_median'])
        histogram_variance = float(request.form['histogram_variance'])
        histogram_tendency = float(request.form['histogram_tendency'])
        
        
        
        
        

        mean_value_of_short_term_variability = float(request.form['mean_value_of_short_term_variability'])
        X = [[ baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_median,histogram_variance,histogram_tendency]]
        X = model1.transform(X)
        output = model.predict(X)

        out = ['Normal', 'Pathological', 'Suspect']
        if int(output[0]) == 0:
            result = 'Normal'
        elif int(output[0]) == 1:
            result = 'Pathological'
        else:
            result = 'Suspect'

        return render_template('output.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)