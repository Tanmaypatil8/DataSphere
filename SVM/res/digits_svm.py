import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()

print(digits.target)
print(dir(digits))

df = pd.DataFrame(digits.data, columns=digits.feature_names)
print(df.head())

df['target'] = digits.target
print(df.head(20))

X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)

# Using RBF kernel

from sklearn.svm import SVC
rbf_model = SVC(kernel='rbf')

rbf_model.fit(X_train, y_train)
print("rbf_kernel",rbf_model.score(X_test, y_test))


# Using Linear kernel

linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)
print("linear_model",linear_model.score(X_test, y_test))