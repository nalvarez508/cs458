from os import pipe
import numpy as np
from sklearn import datasets, tree, model_selection, metrics, preprocessing, pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

# (a) load newsgroups
ng_train = datasets.fetch_20newsgroups(subset='train', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))
ng_test = datasets.fetch_20newsgroups(subset='test', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))

#x1 = ng_train.data
y1 = ng_train.target
#x2 = ng_test.data
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


#grid = model_selection.RandomizedSearchCV(estimator=pipe_, param_distributions=param_grid, scoring='accuracy', refit=True, verbose=True)

# RandomizedSearchCV results.
# {*'clf__min_samples_split': 20, 'clf__min_samples_leaf': 1, 'clf__max_leaf_nodes': None, *'clf__max_depth': 10, 'clf__criterion': 'gini'}

# Optimized results from adjusted param_grid and GridSearchCV.
# {'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__max_leaf_nodes': None, 'clf__min_samples_leaf': 5, 'clf__min_samples_split': 20}

Results_SVC_Penalty = {
  "l1" : 0,
  "l2" : 0
}

Results_Bayes_Alpha = {
  0.001 : 0,
  0.01 : 0,
  0.1 : 0
}

Results_KNN_LeafSize = {
  20 : 0,
  30 : 0,
  40 : 0
}

Results_Ada_LearningRate = {
  0.001 : 0,
  0.01 : 0,
  0.1 : 0,
  1.0 : 0
}

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
  for hp in Results_KNN_LeafSize:
    trainMe(KNeighborsClassifier(n_neighbors=10, leaf_size=hp), Results_KNN_LeafSize, hp)

  # Random forest (RandomForestClassifier)
  #grid = model_selection.GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='accuracy', refit=True, verbose=True)
  #grid.fit(ng_train.data, ng_train.target)


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
print(f'\nK-nearest Neighbors\nHyperparameter: Leaf Size')
printDictReallyNice(Results_KNN_LeafSize)
print(f'\nAdaBoost Classifier\nHyperparameter: Learning Rate')
printDictReallyNice(Results_Ada_LearningRate)

#print(grid.best_params_)
#y_pred = grid.best_estimator_.predict(ng_test.data)
#print(metrics.confusion_matrix(ng_test.target, y_pred))

#tree.plot_tree(grid.best_estimator_['clf'], filled=True, class_names=ng_test.target_names)
#plt.show()