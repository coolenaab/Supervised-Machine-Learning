import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt


print("Inmporting csv")
input_file = r"terrorism-2000-2016-less.csv"
test_perc = 0.40


features = pd.read_csv(input_file, header=0, delimiter=';', usecols=range(11))
labels = pd.read_csv(input_file, header=0, delimiter=';')
labels = labels['gname']

print("Dividing data in test and train sets, using", test_perc * 100, "% as the test set \n")
# split the set in training and test sets with a ratio defined by test_perc
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_perc, random_state=0)


# Create the GaussianNB classifier and fit the training data to train the classifier
start = time.time()
print("Training classifier")
clf = GaussianNB()
clf.fit(features_train, labels_train)
end = time.time()
print("training time:", round(end-start, 3), "s \n")


# Create a vector that contains the predicted labels of the feature test set
# Calculate the accuracy score against the label test set
start = time.time()
print("Calculating accuracy")
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print("Accuracy: ", round((acc * 100), 3), "%")
end = time.time()
print("Accuracy time:", round(end-start, 3), "s \n")


# Create a confusion matrix
start = time.time()
print("Generating confusion matrix")
cnf_matrix = confusion_matrix(labels_test, pred, labels)


# his function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# plot_confusion_matrix #

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, title='Confusion matrix, without normalization')
plt.show()
end = time.time()
print("Confusion matrix time:", round(end-start, 3), "s \n")
