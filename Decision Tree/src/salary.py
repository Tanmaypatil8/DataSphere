import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("Decision Tree/data/salaries.csv")

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

print("Model Score:", model.score(inputs_n, target))

sample_1 = pd.DataFrame([[2, 1, 0]], columns=inputs_n.columns)
sample_2 = pd.DataFrame([[2, 1, 1]], columns=inputs_n.columns)

# Make predictions
print(model.predict(sample_1))  
print(model.predict(sample_2))  
