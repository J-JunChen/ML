import numpy as np
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import random


# Load the dataset
# 选取下面的8类
selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

rand = random.randint(0,9)

newsgroup_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes'), categories=selected_categories, random_state=rand)

# pprint(list(newsgroup_train.target_names))
newsgroup_test = fetch_20newsgroups(
    subset='test', remove=('headers', 'footers', 'quotes'), categories=selected_categories, random_state=rand)

train_data = newsgroup_train['data']
train_label = newsgroup_train['target']
test_data = newsgroup_test['data']
test_label = newsgroup_test['target']

# train_data, train_label = shuffle(train_data, train_label, random_state=3)
# test_data, test_label = shuffle(test_data, test_label, random_state=3)

class Classifier():
    def __init__(self, name, clf):
        self.name = name
        self.classifier = clf

    def classification(self):
        clf = Pipeline(
            [('tfidf', TfidfVectorizer()), ('clf', self.classifier)])
        clf = clf.fit(train_data, train_label)
        pred = clf.predict(test_data)
        print("the Accuracy of %s: %0.4f" %
              (self.name, np.mean(pred == test_label)))
        print(classification_report(test_label, pred))


clf_name = {
            'Naive Bayes': MultinomialNB(alpha=0.01),
            'SVM': SVC(),
            "Linear SVM": LinearSVC(),
            'Logistic Regression': LogisticRegression(),
            'MLP': MLPClassifier(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(n_estimators=8),
            'Adaboost': AdaBoostClassifier()
            }

for key, value in clf_name.items():
    test = Classifier(key, value)
    test.classification()

# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', KNeighborsClassifier())
# ])
# grid_params = {
#     'clf__n_neighbors': [3,5,11,19],
#     'clf__weights': ['uniform', 'distance'],
#     'clf__metric': ['euclidean', 'manhattan']
# }

# gs = GridSearchCV(
#     pipeline,
#     grid_params,
#     verbose=1,
#     cv = 3,
#     n_jobs=-1
# )

# gs_results = gs.fit(train_data, train_label)
# print(gs_results.best_score_)
# print(gs_results.best_estimator_)
# print(gs_results.best_params_)
