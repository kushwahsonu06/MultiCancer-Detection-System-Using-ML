from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('lung_logistic.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float array
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)  # Ensure correct shape for model input

        # Make prediction
        prediction = model.predict(final_features)
        output = 'Lung Cancer' if prediction[0] == 1 else 'No Lung Cancer'
    except Exception as e:
        output = f"Error: {str(e)}"  # Catch and display error if it occurs
        print(f"Prediction error: {e}")  # Print error to console for debugging

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)
