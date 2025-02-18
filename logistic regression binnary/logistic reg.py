import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the data
df = pd.read_csv('logistic regression binnary/data files/HR_comma_sep.csv')

print(df.head)
# Function to handle outliers
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])

# Columns to handle outliers
columns_to_handle_outliers = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
for column in columns_to_handle_outliers:
    handle_outliers(df, column)

# Encode categorical variables using LabelEncoder
department_encoder = LabelEncoder()
salary_encoder = LabelEncoder()
df['Department'] = department_encoder.fit_transform(df['Department'])
df['salary'] = salary_encoder.fit_transform(df['salary'])

# Define features and target variable
X = df.drop('left', axis=1)
y = df['left']

# Scale numerical features
scaler = StandardScaler()
X[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']] = scaler.fit_transform(
    X[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']])

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# # Save the model and encoders
# joblib.dump(model, 'employee_retention_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(department_encoder, 'department_encoder.pkl')
# joblib.dump(salary_encoder, 'salary_encoder.pkl')