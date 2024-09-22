from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # For loading the scaler

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Heart Deploy App!"


# Load the trained model and scaler
model = tf.keras.models.load_model('diabetes_model.h5')  # Path to your model
scaler = joblib.load('scaler.pkl')  # Load the scaler using joblib

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Log the incoming data
    print("Received data:", data)  # Debug line

    # Ensure the input data matches the training data structure
    required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    # Check if the data has the required fields
    if not all(col in data for col in required_columns):
        return jsonify({"error": "Missing required fields"}), 400

    # Create a DataFrame from the input
    input_data = pd.DataFrame(data)

    # Log the DataFrame to check its contents
    print("Input DataFrame:", input_data)  # Debug line

    # Verify if input_data is not empty
    if input_data.empty:
        return jsonify({"error": "Input data is empty"}), 400

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_scaled)

    # Convert prediction to a response format
    response = {
        'probability': float(prediction[0][0]),  # Convert to native Python float
        'recommendation': get_recommendation(prediction[0][0])
    }
    return jsonify(response)

def get_recommendation(pred):
    if pred >= 0.7:
        return "The predicted probability of diabetes is high. It's advised to see a doctor immediately."
    elif pred >= 0.5:
        return "The predicted probability of diabetes is moderate. It's recommended to schedule a checkup."
    else:
        return "The predicted probability of diabetes is low. Everything is fine!"

if __name__ == '__main__':
    app.run(debug=True)
