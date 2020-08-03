import warnings
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('./data/heart_disease.csv', header=None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
sc = ss()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


class Classifier():
    def __init__(self, name, clf):
        self.name = name
        self.clf = clf

    def classification(self):
        classifier = self.clf
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print("The Accuracy of %s: %.04f" %
              (self.name, accuracy_score(y_test, y_pred)))
        print(classification_report(y_test, y_pred))


model = {
    'SVM': SVC(kernel='rbf'),
    'Linear SVM': LinearSVC(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

for key, value in model.items():
    clf = Classifier(key, value)
    clf.classification()
