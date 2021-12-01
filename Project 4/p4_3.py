from sklearn import datasets, metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

x1,y = datasets.load_digits(return_X_y=True)
p = PCA(2)
p.fit(x1)
x = p.transform(x1)
print(f"Dimensionality reduced from {x1.shape[1]} to {x.shape[1]}")

# Cluster the data
_kmeans = [
  ("km_10cluster_.00001", KMeans(n_clusters=10, tol=1e-5)),
  ("km_12cluster_.1", KMeans(n_clusters=12, tol=1e-1)),
  ("km_10cluster_.01", KMeans(n_clusters=10, tol=1e-2)),
  ("km_9cluster_.0001", KMeans(n_clusters=9, tol=1e-4))
]

_dbscan = [
  ("db_1sample_1.1", DBSCAN(min_samples=1, eps=1.1)),
  ("db_1sample_1.2", DBSCAN(min_samples=1, eps=1.2)),
  ("db_1sample_1.3", DBSCAN(min_samples=1, eps=1.3))
]

def helper_dbTuning():
  for i in range(7, 21): #eps
    i10 = i/10.0
    tempTuple = [(i10, DBSCAN(min_samples=2, eps=i10))]
    runModel(tempTuple)

def helper_kMeansTuning():
  for i in range(2, 13): #n_clusters
    for j in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]: #tolerance
      tempTuple = [(f"s{i} tol{j}", KMeans(n_clusters=i, tol=j))]
      runModel(tempTuple)

def runModel(m):
  for name, est in m:
    est.fit_predict(x)
    score = metrics.adjusted_rand_score(y, est.labels_)
    print(f"{name}: \t{score}")

print("Random Index Adjusted for Chance\n(closer to 1.0 is better)\n")
#helper_kMeansTuning()
#helper_dbTuning()
runModel(_kmeans)
print()
runModel(_dbscan)