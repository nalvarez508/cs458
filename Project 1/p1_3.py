import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets

# import the data
iris = datasets.load_iris()
x = iris.data
y = iris.target
IrisLabels = ['Versicolor', 'Setosa', 'Virginica']
PlotColors = ['r', 'b', 'g']

fig, axs = plt.subplots()
fig2, axs2 = plt.subplots()

# No decomposition
def generateBasePlot():
  axs.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.brg)
  axs.set_xlabel("Sepal Length (cm)")
  axs.set_ylabel("Sepal width (cm)")

# Decompostion
def generatePCAPlot():
  pca_iris = PCA(3)
  pca_iris.fit(x)
  decomp_x = pca_iris.transform(x)
  axs2.scatter(decomp_x[:,0], decomp_x[:,1], c=y, cmap=plt.cm.brg)
  axs2.set_xlabel("Sepal Length (cm)")
  axs2.set_ylabel("Sepal width (cm)")

generateBasePlot()
generatePCAPlot()

leg = []
for l in range(len(IrisLabels)):
  leg.append(Rectangle((0,0),1,1,fc=PlotColors[l]))
axs.legend(leg, IrisLabels)
axs2.legend(leg, IrisLabels)


plt.show()