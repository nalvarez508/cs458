import matplotlib.pyplot as plt
from sklearn import datasets

NUMBER_PLOTS_SQ = 4

# import the data
iris = datasets.load_iris()
x = iris.data
y = iris.target
TitleList = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

fig, axs = plt.subplots(NUMBER_PLOTS_SQ, NUMBER_PLOTS_SQ)
fig.suptitle("Iris Data (red=setosa, green=versicolor, blue=virginica)")

def generatePlots():
  for v in range(NUMBER_PLOTS_SQ):
    for h in range(NUMBER_PLOTS_SQ):
      if (v != h):
        axs[v][h].scatter(x[:, h], x[:, v], s=7, c=y, cmap=plt.cm.brg, edgecolor='k', linewidth=0.5)
      else:
        axs[v][h].text(0.5, 0.5, TitleList[v], horizontalalignment='center', verticalalignment='center', clip_on=True)
        axs[v][h].xaxis.set_visible(False)
        axs[v][h].yaxis.set_visible(False)

generatePlots() #(a)
plt.show()