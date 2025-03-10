import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Random_Forest/data/loan.csv")

# Set the style to a valid matplotlib style
plt.style.use('seaborn-v0_8')  # or use 'ggplot'
sns.set_theme(style="whitegrid")  # Set seaborn style

# Create figure for first set of plots
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Loan_Status', palette='Set2')
plt.title('Loan Status Distribution', pad=20)

plt.subplot(2, 2, 2)
sns.histplot(data=df, x='LoanAmount', bins=30, color='skyblue')
plt.title('Loan Amount Distribution', pad=20)

plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', palette='Set2')
plt.title('Income vs Loan Amount', pad=20)

plt.subplot(2, 2, 4)
sns.countplot(data=df, x='Education', hue='Loan_Status', palette='Set2')
plt.title('Loan Status by Education', pad=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ...rest of your existing code...

plt.figure(figsize=(10, 8))
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
correlation = df[numerical_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

categorical_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for column in categorical_columns:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)

numerical_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for column in numerical_columns:
    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

for column in numerical_columns:
    df = remove_outliers_iqr(df, column)

X = df.drop(columns=['Loan_Status'])  
y = df['Loan_Status']  

X_encoded = pd.get_dummies(X)

print("Class distribution before resampling:")
print(y.value_counts())

oversampler = RandomOverSampler(random_state=42)
X_over, y_over = oversampler.fit_resample(X_encoded, y)

print("\nClass distribution after oversampling:")
print(y_over.value_counts())

undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_encoded, y)

print("\nClass distribution after undersampling:")
print(y_under.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)