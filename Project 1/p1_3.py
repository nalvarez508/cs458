import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets

# import the data
iris = datasets.load_iris()
x = iris.data
y = iris.target

fig, axs = plt.subplots()
fig2, axs2 = plt.subplots()

# No decomposition
def generateBasePlot():
  axs.scatter(x[:, 0], x[:, 1])
  axs.set_xlabel("Sepal Length (cm)")
  axs.set_ylabel("Sepal width (cm)")

# Decompostion
def generatePCAPlot():
  pca_iris = PCA(3)
  pca_iris.fit(x)
  decomp_x = pca_iris.transform(x)
  axs2.scatter(decomp_x[:,0], decomp_x[:,1])
  axs2.set_xlabel("Sepal Length (cm)")
  axs2.set_ylabel("Sepal width (cm)")

generateBasePlot()
generatePCAPlot()
plt.show()