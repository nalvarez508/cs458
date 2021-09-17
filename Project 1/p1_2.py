import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

# Create Figure 1
fig, axs = plt.subplots(NUMBER_PLOTS_SQ, NUMBER_PLOTS_SQ)
fig.suptitle("Iris Data (blue=setosa, red=versicolor, green=virginica)")
# Create Figure 2
fig2, axs2 = plt.subplots()

# Iris 16 plots
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
  # Scatter two variables
  IrisLabels = ['Setosa', 'Versicolor', 'Virginica']
  for i in range(np.prod(y.shape)):
    axs2.scatter(x[i, 2], x[i, 3], marker=MarkerList[int(i/50)], c='k', s=10.0)
  #axs2.set_label(['Setosa', 'Versicolor', 'Virginica'])
  for l in range(len(IrisLabels)):
    axs2.scatter([],[],color='k', label=IrisLabels[l], marker=MarkerList[l])
  axs2.legend()
  
  # Delete unnecessary variables
  ClusterArray = np.delete(x, [0,1], 1)

  # Generate clusters
  axs2.set_xlabel("Petal Length (cm)")
  axs2.set_ylabel("Petal Width (cm)")
  km = KMeans(n_clusters=3)
  km.fit(ClusterArray)

  # Find centroids and plot them
  km_cntr = km.cluster_centers_
  axs2.scatter(km_cntr[:, 0], km_cntr[:, 1], c='red', s=15.0)
  
  # Create rectangles around clusters
  CentroidRectangleLengths = np.empty((0,2))
  for i in range(0,101,50):
    try:
      val_min_x, val_max_x = np.min(np.delete(ClusterArray[i:(i+50)], 1, 1), axis=0), np.max(np.delete(ClusterArray[i:(i+50)], 1, 1), axis=0)
      val_min_y, val_max_y = np.min(np.delete(ClusterArray[i:(i+50)], 0, 1), axis=0), np.max(np.delete(ClusterArray[i:(i+50)], 0, 1), axis=0)
      x_len, y_len = val_max_x-val_min_x, val_max_y-val_min_y
      CentroidRectangleLengths = np.append(CentroidRectangleLengths, [np.concatenate((x_len, y_len))], axis=0)
    except ValueError:
      pass
  
  # Plot the rectangles
  index=0
  sortedCenters = np.sort(km_cntr, axis=0)
  for c in (sortedCenters):
    width, height = CentroidRectangleLengths[index][0], CentroidRectangleLengths[index][1]
    axs2.add_patch(Rectangle(
      xy=(c[0]-width/2, c[1]-height/2), width=width, height=height, linewidth=1, color='black', fill=False))
    index += 1

generatePlots() #(a)
generateDiscretization() #(b)
plt.show()