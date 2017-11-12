import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# Load iris dataset and fit it for a linear support vector
iris = datasets.load_iris()
print(iris)

output = iris.data.shape, iris.target.shape
print(output)

# Sample a training set while holding 40% of the data for testing the classifier
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

output = X_train.shape, y_train.shape
print(output)

output = X_test.shape, y_test.shape
print(output)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
output = clf.score(X_test, y_test)
print(output)

