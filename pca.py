import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.datasets import __all__
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Implement logistic regression for prediction (optional)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print the names of all datasets
for dataset_name in __all__:
    print(dataset_name)

# Evaluate the model's accuracy (optional)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

