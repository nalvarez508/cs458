from os import truncate
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

cluster_types = ["ward", "complete", "average", "single"]

# (a) Generate data points
np.random.seed(0)
x1 = np.random.randn(50,2)+[2,2]
x2 = np.random.randn(50,2)+[6,10]
x3 = np.random.randn(50,2)+[10,2]
x = np.concatenate((x1,x2,x3))

# (b) Cluster and plot points
fig, axs = plt.subplots(nrows=2, ncols=2, )

def createDendogram(model):
  counts = np.zeros(model.children_.shape[0])
  n_samples = len(model.labels_)
  for i, m in enumerate(model.children_):
    current_count = 0
    for child_idx in m:
      if child_idx < n_samples:
        current_count += 1  # leaf node
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count

  return np.column_stack([model.children_, model.distances_, counts]).astype(float)


def plotClusters():
  clusterIndex = 0
  for row in range(0,2):
    for col in range(0,2):
      _cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage=cluster_types[clusterIndex]).fit(x)
      linkage_matrix = createDendogram(_cluster)
      dendrogram(linkage_matrix, ax=axs[row, col], truncate_mode="level", p=3, no_labels=True)
      plt.title(cluster_types[clusterIndex])
      axs[row, col].set_title(cluster_types[clusterIndex].upper())
      clusterIndex += 1

plotClusters()
fig.suptitle("P4-1")
plt.show()