import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

NUMBER_PLOTS_SQ = 4

# import the data
iris = datasets.load_iris()
x = iris.data
y = iris.target

MarkerList = ['s', 'o', 'd']
TitleList = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]


fig, axs = plt.subplots(NUMBER_PLOTS_SQ, NUMBER_PLOTS_SQ)
fig.suptitle("Iris Data (blue=setosa, red=versicolor, green=virginica)")
fig2, axs2 = plt.subplots()

def generatePlots():
  for v in range(NUMBER_PLOTS_SQ):
    for h in range(NUMBER_PLOTS_SQ):
      if (v != h):
        axs[v][h].scatter(x[:, h], x[:, v], s=7, c=y, cmap=plt.cm.brg, edgecolor='k', linewidth=0.5)
      else:
        axs[v][h].text(0.5, 0.5, TitleList[v], horizontalalignment='center', verticalalignment='center', clip_on=True)
        axs[v][h].xaxis.set_visible(False)
        axs[v][h].yaxis.set_visible(False)

def generateDiscretization():
  for i in range(np.prod(y.shape)):
    axs2.scatter(x[i, 2], x[i, 3], marker=MarkerList[int(i/50)], c='k', s=10.0)
  
  ClusterArray = np.delete(x, [0,1], 1)
  print(ClusterArray)

  axs2.set_xlabel("Petal Length")
  axs2.set_ylabel("Petal Width")
  km = KMeans(n_clusters=3)
  km.fit(ClusterArray)

  km_cntr = km.cluster_centers_
  axs2.scatter(km_cntr[:, 0], km_cntr[:, 1], c='red', s=15.0)

generatePlots() #(a)
generateDiscretization() #(b)
plt.show()