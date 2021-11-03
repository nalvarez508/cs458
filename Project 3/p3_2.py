from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# (a) build classifier
digits = datasets.load_digits()
x1, x2, y1, y2 = train_test_split(digits.data, digits.target, test_size=0.5)
good_clf = None

class Neural:
  def __init__(self, a):
    self.alpha = a
    self.clf = MLPClassifier(alpha=a)
    self.score = 0
  
  def update(self, s):
    self.score = s

Results_MLP_Alpha = {
  0.0001 : Neural(0.0001),
  0.001 : Neural(0.001),
  0.01 : Neural(0.01),
  0.1 : Neural(0.1)
}

Results_SVC_Gamma = {
  0.0001 : 0,
  0.001 : 0,
  0.01 : 0,
  0.1 : 0
}

def trainMe(clf, Results, hyperparameter):
  print(f"Testing {str(clf)}")
  clf.fit(x1, y1)
  pred = clf.predict(x2)
  score = metrics.accuracy_score(y2, pred)
  try:
    Results[hyperparameter].score = score
  except AttributeError:
    Results[hyperparameter] = score

def runTests():
  global good_clf

  # MLPClassifier
  max_score = 0
  for hp in Results_MLP_Alpha:
    trainMe(Results_MLP_Alpha[hp].clf, Results_MLP_Alpha, hp)
    if Results_MLP_Alpha[hp].score > max_score:
      max_score = Results_MLP_Alpha[hp].score
      good_clf = Results_MLP_Alpha[hp].clf

  # SVC
  for hp in Results_SVC_Gamma:
    trainMe(SVC(gamma=hp, tol=1e-3), Results_SVC_Gamma, hp)

def printDictReallyNice(d):
  for k,v in d.items():
    try:
      print(k, ' : ', v.score)
    except AttributeError:
      print(k, ' : ', v)

runTests()
print("\nFormat = Hyperparameter : Accuracy")
print(f'\nMulti-layer Perceptron\nHyperparameter: Regularization (alpha)')
printDictReallyNice(Results_MLP_Alpha)
print(f'\nSupport Vector Machine\nHyperparameter: Gamma')
printDictReallyNice(Results_SVC_Gamma)
print()
print(f"Using {str(good_clf)} as best NN...\n")
print(" -- Confusion Matrix of Neural Network --")
print(metrics.confusion_matrix(y2, good_clf.predict(x2)))