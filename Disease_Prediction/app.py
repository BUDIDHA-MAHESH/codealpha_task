from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model, label_encoder, feature_names = pickle.load(open("models/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', symptoms=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    input_symptoms = request.form.getlist('symptoms')
    input_vector = [1 if feature in input_symptoms else 0 for feature in feature_names]
    prediction = model.predict([input_vector])[0]
    disease = label_encoder.inverse_transform([prediction])[0]
    return render_template('result.html', prediction=disease)

if __name__ == "__main__":
    app.run(debug=True)
