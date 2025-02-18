from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and label encoders
model = joblib.load('employee_retention_model.pkl')
scaler = joblib.load('scaler.pkl')
department_encoder = joblib.load('department_encoder.pkl')
salary_encoder = joblib.load('salary_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.form.to_dict()
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Convert categorical variables to numerical values
        df['Department'] = department_encoder.transform(df['Department'])
        df['salary'] = salary_encoder.transform(df['salary'])
        
        # Convert all other columns to float
        for col in df.columns:
            if col not in ['Department', 'salary']:
                df[col] = df[col].astype(float)
        
        # Scale numerical features
        df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']] = scaler.transform(
            df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']])
        
        # Predict the result
        prediction = model.predict(df)
        
        # Interpret the prediction
        result = "leave the company" if prediction[0] == 1 else "stay with the company"
        
        # Render the result page
        return render_template('result.html', prediction=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)