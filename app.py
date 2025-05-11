from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model_path = "my_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(request.form[f]) for f in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal'
        ]]
        input_data = np.array([features])
        
        # Predict
        prediction = model.predict(input_data)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
