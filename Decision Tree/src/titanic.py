import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix

df = pd.read_csv("Decision Tree/data/titanic.csv")
print(df.head())

inputs = df.drop(['Name', 'PassengerId', 'Survived', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
target = df['Survived']

inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())

le_Sex = LabelEncoder()
inputs['Sex'] = le_Sex.fit_transform(inputs['Sex'])

print(inputs.head())

model = tree.DecisionTreeClassifier()
model.fit(inputs, target)


predictions = model.predict(inputs)
print("Predictions:\n", predictions)
cm = confusion_matrix(target, predictions)
print("Confusion Matrix:\n", cm)


print("Model Score:", model.score(inputs, target))
