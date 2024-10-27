# app.py

import pickle
from flask import Flask, request, jsonify

# Define model and vectorizer files
model_file = 'model1.bin'
vectorizer_file = 'dv.bin'

# Load DictVectorizer and model
with open(vectorizer_file, 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open(model_file, 'rb') as f_model:
    model = pickle.load(f_model)

# Initialize Flask app
app = Flask('subscription')

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get client data from the request in JSON format
    client_data = request.get_json()
    
    # Transform the input data and make prediction
    X = dv.transform([client_data])
    y_pred = model.predict_proba(X)[0, 1]
    
    # Return the result as JSON
    result = {
        'subscription_probability': float(y_pred),
        'subscription': bool(y_pred >= 0.5)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)