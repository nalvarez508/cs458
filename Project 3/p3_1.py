import numpy as np
from sklearn import datasets, model_selection, metrics
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# (a) load newsgroups
ng_train = datasets.fetch_20newsgroups(subset='train', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))
ng_test = datasets.fetch_20newsgroups(subset='test', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))

y1 = ng_train.target
y2 = ng_test.target

# (b) classifiers

max_depth_range = [None, 2, 5, 10]
min_samples_leaf_range = [1, 5, 10]
min_sample_split_range = [2, 10, 20]
min_leaf_nodes_range = [None, 5, 10, 20]

param_grid = {"clf__criterion": ['gini'],
              "clf__max_depth": [10],
              "clf__min_samples_leaf": [1, 5, 10],
              "clf__min_samples_split": [20],
              "clf__max_leaf_nodes": [None, 5, 10, 20]
              }

pipe_rf = Pipeline([('vect', TfidfVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', RandomForestClassifier())])

Results_SVC_Penalty = {
  "l1" : 0,
  "l2" : 0
}

Results_Bayes_Alpha = {
  0.001 : 0,
  0.01 : 0,
  0.1 : 0
}

Results_KNN_Neighbors = {
  5 : 0,
  10 : 0,
  15 : 0
}

Results_Ada_LearningRate = {
  0.001 : 0,
  0.01 : 0,
  0.1 : 0,
  1.0 : 0
}

Results_Forest_Multiple = dict()

def trainMe(clf, Results, hyperparameter):
  print(f"Testing {str(clf)}")
  clf.fit(x1, y1)
  pred = clf.predict(x2)
  score = metrics.accuracy_score(y2, pred)
  Results[hyperparameter] = score

def runTests():
  # Support Vector Machine (LinearSVC)
  for hp in Results_SVC_Penalty:
    trainMe(LinearSVC(penalty=hp, tol=1e-3, dual=False), Results_SVC_Penalty, hp)

  # Naive Bayes (MultinomialNB)
  for hp in Results_Bayes_Alpha:
    trainMe(MultinomialNB(alpha=hp), Results_Bayes_Alpha, hp)

  # K-nearest Neighbors (KNeighborsClassifier)
  for hp in Results_KNN_Neighbors:
    trainMe(KNeighborsClassifier(n_neighbors=hp), Results_KNN_Neighbors, hp)

  # Random forest (RandomForestClassifier)
  print("Testing RandomForestClassifier(*)")
  grid = model_selection.GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='accuracy', refit=True, verbose=True)
  grid.fit(ng_train.data, ng_train.target)
  means = grid.cv_results_["mean_test_score"]
  for mean, params in zip(means, grid.cv_results_["params"]):
    Results_Forest_Multiple.update({str(params) : mean})

  # AdaBoost (AdaBoostClassifier)
  for hp in Results_Ada_LearningRate:
    trainMe(AdaBoostClassifier(learning_rate=hp), Results_Ada_LearningRate, hp)
  
def printDictReallyNice(d):
  for k,v in d.items():
    print(k, ' : ', v)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english',)
x1 = vectorizer.fit_transform(ng_train.data)
x2 = vectorizer.transform(ng_test.data)

runTests()
print("\nFormat = Hyperparameter : Accuracy")
print(f'\nSupport Vector Machine\nHyperparameter: Penalty')
printDictReallyNice(Results_SVC_Penalty)
print(f'\nNaive Bayes\nHyperparameter: Smoothing (alpha)')
printDictReallyNice(Results_Bayes_Alpha)
print(f'\nK-nearest Neighbors\nHyperparameter: Number of Neighbors')
printDictReallyNice(Results_KNN_Neighbors)
print(f'\nRandom Forest\nHyperparameter: Max Depth, Min Samples Leaf, Min Samples Split, Min Leaf Nodes')
printDictReallyNice(Results_Forest_Multiple)
print(f'\nAdaBoost Classifier\nHyperparameter: Learning Rate')
printDictReallyNice(Results_Ada_LearningRate)