import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_json("C:/Users/tanma/PycharmProjects/dataset/train-whats-cooking.json")
print(df.head())

one_hot_encoded = pd.get_dummies(df['cuisine'], prefix='Cuisine')

df_encoded = pd.concat([df, one_hot_encoded], axis=1)

df_encoded.drop('cuisine', axis=1, inplace=True)

X = df_encoded.drop(['id', 'ingredients'], axis=1)
y = df['cuisine']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred, labels=df['cuisine'].unique())

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['cuisine'].unique(), yticklabels=df['cuisine'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
