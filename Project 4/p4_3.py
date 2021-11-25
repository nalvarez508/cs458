from types import LambdaType
from sklearn import datasets, metrics
from sklearn.cluster import DBSCAN, KMeans

x,y = datasets.load_digits(return_X_y=True)

# Cluster the data
_kmeans = [
  ("km_8cluster_4", KMeans(n_clusters=8, tol=1e-4)),
  ("km_4cluster_4", KMeans(n_clusters=4, tol=1e-4)),
  ("km_8cluster_3", KMeans(n_clusters=8, tol=1e-3)),
  ("km_4cluster_3", KMeans(n_clusters=4, tol=1e-3))]

_dbscan = [
  ("db_3", DBSCAN(eps=0.3)),
  ("db_5", DBSCAN(eps=0.5)),
  ("db_7", DBSCAN(eps=0.7))
]

def runModel(m):
  for name, est in m:
    est.fit(x)
    score = metrics.adjusted_rand_score(y, est.labels_)
    print(f"{name}: {score}")

runModel(_kmeans)
runModel(_dbscan)