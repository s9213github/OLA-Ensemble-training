from flask import Flask, request, jsonify
import pickle
import sklearn
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('Ola_Model.pkl', 'rb') as f:
    model,scaler,dummy_columns = pickle.load(f)
    
@app.route('/')
def home():
    return "Welcome to the OLA Model Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame({
        'Driver_ID': [data['Driver_ID']],
        'Age': [data['Age']],
        'Gender': [data['Gender']],
        'Education_Level': [data['Education_Level']],
        'Income': [data['Income']],
        'Grade': [data['Grade']],
        'Total Business Value': [data['Total Business Value']],
        "Promoted": [data['Promoted']],
        'Rating_Change': [data['Rating_Change']],
        'Income_Change': [data['Income_Change']],
    })
    input_data = pd.get_dummies(input_data, columns=['Rating_Change','Income_Change'], drop_first=True)
    input_data = input_data.reindex(columns=dummy_columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    # Ensure the input data has the same columns as the model expects
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=dummy_columns)
    prediction = model.predict(input_data_scaled)
    prediction = model.predict(input_data_scaled)
    result = int(prediction[0])
    if result == 1:
        return jsonify({'churn_prediction': 'Driver-partner is likely to churn.'})
    else:
        return jsonify({'churn_prediction': 'Driver-partner is not likely to churn.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)