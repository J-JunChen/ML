import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.color import rgb2grey

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import time

train_data = scipy.io.loadmat('./data/train_32x32.mat')
test_data = scipy.io.loadmat('./data/test_32x32.mat')

X_train = train_data['X']
X_test = test_data['X']
y_train = train_data['y']
y_test = test_data['y']

# flatten the picture
X_train = X_train.reshape(
    X_train.shape[0]*X_train.shape[1]*X_train.shape[2], X_train.shape[3]).T
y_train = y_train.reshape(y_train.shape[0],)

X_test = X_test.reshape(
    X_test.shape[0]*X_test.shape[1]*X_test.shape[2], X_test.shape[3]).T
y_test = y_test.reshape(y_test.shape[0],)
print(X_train.shape)
print(y_train.shape)

X_train, y_train = shuffle(X_train, y_train, random_state=3)
X_test,  y_test = shuffle(X_test, y_test, random_state=3)

# X_train = X_train[:3000, :]
# y_train = y_train[:3000]
# X_test = X_test[:500, :]
# y_test = y_test[:500]
X_train = X_train[:10000, :]
y_train = y_train[:10000]
X_test = X_test[:2000, :]
y_test = y_test[:2000]

# X_train = rgb2grey(X_train)
# y_train = rgb2grey(y_train)
# plt.imshow(X_train[1])
# plt.show()


class Classifier():
    def __init__(self, name, clf):
        self.name = name
        self.clf = clf

    def classify(self):
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print("The Accuracy of % s: % 0.4f " %
              (self.name, accuracy_score(y_test, y_pred)))
        print(classification_report(y_test, y_pred))


clf_name = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='poly'),
    "Linear SVM": LinearSVC(),
    'Logistic Regression': LogisticRegression(),
    'MLP': MLPClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Adaboost': AdaBoostClassifier()
}

for i in range(1):
    t1 = time.time()
    for key, value in clf_name.items():
        clf = Classifier(key, value)
        clf.classify()
    t2 = time.time()
    print("The code run {:.0f}m {:.0f}s".format((t2-t1)//60, (t2-t1) % 60))
