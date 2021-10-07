from os import pipe
import numpy as np
from sklearn import datasets, tree, model_selection, metrics, preprocessing, pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt

# (a) load newsgroups
ng_train = datasets.fetch_20newsgroups(subset='train', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))
ng_test = datasets.fetch_20newsgroups(subset='test', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))

print(f'''
Set\t_|_ # Docs\t_|_ Attributes''')

# (b) decision tree

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

pipe_ = Pipeline([('vect', TfidfVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', tree.DecisionTreeClassifier())])

#grid = model_selection.RandomizedSearchCV(estimator=pipe_, param_distributions=param_grid, scoring='accuracy', refit=True, verbose=True)

# RandomizedSearchCV results.
# {*'clf__min_samples_split': 20, 'clf__min_samples_leaf': 1, 'clf__max_leaf_nodes': None, *'clf__max_depth': 10, 'clf__criterion': 'gini'}

# Optimized results from adjusted param_grid and GridSearchCV.
# {'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__max_leaf_nodes': None, 'clf__min_samples_leaf': 5, 'clf__min_samples_split': 20}

grid = model_selection.GridSearchCV(estimator=pipe_, param_grid=param_grid, scoring='accuracy', refit=True, verbose=True)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english',)
x_train = vectorizer.fit_transform(ng_train.data)

print(f'''Train\t |  {ng_train.target.shape[0]}\t |  {x_train.shape[1]}
Test\t |  {ng_test.target.shape[0]}\t |  N/A''')

grid.fit(ng_train.data, ng_train.target)
print(grid.best_params_)
y_pred = grid.best_estimator_.predict(ng_test.data)
print(metrics.confusion_matrix(ng_test.target, y_pred))

#tree.plot_tree(grid.best_estimator_['clf'], filled=True, class_names=ng_test.target_names)
#plt.show()