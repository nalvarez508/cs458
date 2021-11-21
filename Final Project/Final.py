import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from Class_Solar import Solar

NUMBER_ATTRIBUTES = 15
BEGIN_AT = 0
_VALUETOTEST = 8
_NUMBERZONES = 3
np.set_printoptions(precision=5, suppress=True)

s_train = Solar("solar_training.csv")
print("Shape of training data:", s_train.data.shape)
s_test = Solar("solar_test.csv")#, 48159)
print("Shape of test data:", s_test.data.shape)

RMSE_Scores = [0,0,0]
MAE_Scores = [0,0,0]

def truncate(num, digits):
  l = str(float(num)).split('.')
  digits = min(len(l[1]), digits)
  return l[0] + '.' + l[1][:digits]

def generatePlots():
  fig, axs = plt.subplots(NUMBER_ATTRIBUTES, NUMBER_ATTRIBUTES)
  fig.tight_layout()
  plt.scatter(s_train.var78, s_train.power, s=0.1)

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

def makeModel():
  model = svm.SVR()
  model.fit(s_train.data, s_train.power)
  y_pred_train = model.predict(s_train.data)
  #print(f"RMSE with training data: {metrics.mean_squared_error(s_train.power, y_pred_train, squared=False)}%")
  return model

def runModel():
  for z in range(0, _NUMBERZONES):
    y_pred_test = regr.predict(s_test.zonedata[z])
    #print(f"RMSE of test data on Zone {z}: {metrics.mean_squared_error(s_test.zonepower[z], y_pred_test, squared=False)}%")
    RMSE_Scores[z] = metrics.mean_squared_error(s_test.zonepower[z], y_pred_test, squared=False)
    MAE_Scores[z] = metrics.mean_absolute_error(s_test.zonepower[z], y_pred_test)

def printScores():
  RMSE_out = str()
  MAE_out = str()
  RMSE_avg = 0
  MAE_avg = 0
  for i in range(3):
    RMSE_out += ("\t" + truncate(RMSE_Scores[i], 6))
    RMSE_avg += RMSE_Scores[i]
    MAE_out += ("\t" + truncate(MAE_Scores[i], 6))
    MAE_avg += MAE_Scores[i]
  RMSE_out += ("\t" + truncate(RMSE_avg/3.0, 6))
  MAE_out += ("\t" + truncate(MAE_avg/3.0, 6))

  print("\n\t\t###### Scoring Metrics ######")
  print("\tZone 1\t\tZone 2\t\tZone 3\t\tOverall")
  print(f"RMSE{RMSE_out}")
  print(f"MAE{MAE_out}")
  

regr = makeModel()
runModel()
printScores()