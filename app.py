from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load model and scaler
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Titanic Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Required features
    required = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if not all(key in data for key in required):
        return jsonify({"error": f"Missing fields. Required: {required}"}), 400

    try:
        input_df = pd.DataFrame([data])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0].tolist()

        return jsonify({
            "prediction": int(pred),
            "probabilities": proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
