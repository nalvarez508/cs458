import numpy as np
import matplotlib.pyplot as plt
from Class_Solar import Solar
NUMBER_ATTRIBUTES = 15
BEGIN_AT = 0
s_train = Solar("solar_training.csv")

fig, axs = plt.subplots(NUMBER_ATTRIBUTES, NUMBER_ATTRIBUTES)
fig.tight_layout()
#plt.scatter(s_train.var78, s_train.power, s=0.1)

def generatePlots():
  attributes = s_train.getAttributes()
  names = s_train.getNames()
  coef = s_train.correlation()
  #print(attributes[0].dtype.names[0])
  #print("Sus=",attributes[0])
  for v in range(BEGIN_AT, NUMBER_ATTRIBUTES):
    for h in range(BEGIN_AT, NUMBER_ATTRIBUTES):
      if (v != h):
        axs[v][h].scatter(attributes[h], attributes[v], s=0.1, c=s_train.power, cmap=plt.cm.Greys)
        axs[v][h].xaxis.set_visible(False)
        axs[v][h].yaxis.set_visible(False)
        if v >= 2 and h>=2:
          xmin, xmax, ymin, ymax = axs[v][h].axis()
          xbar = (abs(xmax)-abs(xmin))/2.
          ybar = (abs(ymax)-abs(ymin))/2.
          axs[v][h].text(xbar, ybar, "{:.2f}".format(coef[v-2,h-2]), c='red', horizontalalignment='center', verticalalignment='center', clip_on=True)
      else:
        axs[v][h].text(0.5, 0.5, names[v], horizontalalignment='center', verticalalignment='center', clip_on=True)
        axs[v][h].xaxis.set_visible(False)
        axs[v][h].yaxis.set_visible(False)

generatePlots()
plt.show()

#s_train.correlation()