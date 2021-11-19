from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import kneighbors_graph
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax3d

# (a) generate swiss roll dataset
n_samples = 1500
noise = 0.05
x, _ = datasets.make_swiss_roll(n_samples, noise=noise)
x[:, 1] *= .5

# (b) Agglomerative Clustering
connectivity = kneighbors_graph(x, n_neighbors=10, include_self=False)
_ag_cluster = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward').fit(x)
fig = plt.figure()
ax = ax3d.Axes3D(fig)
ax.view_init(7, -80)
ag_labels = _ag_cluster.labels_
for l in np.unique(ag_labels):
  ax.scatter(x[ag_labels==l, 0], x[ag_labels==l, 1], x[ag_labels==l,2], edgecolor='k')
ax.set_title("Agglomerative Clustering with KNN Connectivity Graph")

# (c) DBSCAN
_db_cluster = DBSCAN().fit(x)
fig2 = plt.figure()
ax2 = ax3d.Axes3D(fig2)
ax2.view_init(7, -80)
db_labels = _db_cluster.labels_
for l in np.unique(db_labels):
  ax2.scatter(x[db_labels==l, 0], x[db_labels==l, 1], x[db_labels==l,2], edgecolor='k')
ax2.set_title("DBSCAN Clustering")

plt.title("HW4-2")
plt.show()