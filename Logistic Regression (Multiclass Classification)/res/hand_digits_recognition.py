from sklearn.datasets import load_digits 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
digits = load_digits()

# Visualize some samples
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

print("Sample Targets:", digits.target[:5])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
modle = LogisticRegression(max_iter=500)
modle.fit(X_train, y_train)

# Evaluate model
print("Accuracy:", modle.score(X_test, y_test))
y_predicted = modle.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
