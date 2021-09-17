import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from progressbar import progressbar

MAX_ITERATIONS = 10
NUMBER_OF_POINTS = 50

## 1000 points under two dimension
#points = np.random.rand(1000,2)
#print(points)

def findMinMax(p):
  hull = ConvexHull(p)
  # Extract the points forming convex the hull
  hullpoints = p[hull.vertices,:]
  # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
  hdist = cdist(hullpoints, hullpoints, metric='euclidean')
  
  ### This code performs as well as cdist. distance.euclidian failed me
    #b = hullpoints.reshape(hullpoints.shape[0], 1, hullpoints.shape[1])
    #hdist = np.sqrt(np.einsum('ijk, ijk->ij', hullpoints-b, hullpoints-b))
  #######

  # Get the min/max points
  maxpair = np.unravel_index(hdist.argmax(), hdist.shape)
  i,j = np.where(hdist==np.min(hdist[np.nonzero(hdist)]))

  pairPoints = np.array([(np.linalg.norm(hullpoints[i][0] - hullpoints[i][1])),(np.linalg.norm(hullpoints[maxpair[0]] - hullpoints[maxpair[1]]))])
  logDiff = np.log10((pairPoints[1]-pairPoints[0])/pairPoints[0])
  print(logDiff)
  listOfLogDiffs.append(logDiff)

## Plotting the log results
def plotGraph():
  x = np.arange(2,MAX_ITERATIONS)
  y = listOfLogDiffs
  plt.title("Curse of Dimensionality") 
  plt.xlabel("Number of Dimensions") 
  plt.ylabel("Log_10 * ((Max-Min)/Min)") 
  plt.plot(x,y) 
  plt.show()

listOfLogDiffs = []
#bigList = []
#averagePoints = []

#for y in progressbar(range(2,MAX_ITERATIONS), redirect_stdout=True):
#  listOfLogDiffs = []
for x in progressbar(range(2,MAX_ITERATIONS), redirect_stdout=True):
  print(f"({x}/{MAX_ITERATIONS}): ", end='')
  findMinMax(np.random.rand(NUMBER_OF_POINTS,x))
  #bigList.append(listOfLogDiffs)


## Multiple Runs // Average the points for each dimension
#print(listOfLogDiffs)
#for x in range(0, len(bigList)):
#  pointsToSum = 0.0
#  for y in range(0, MAX_ITERATIONS):
#    pointsToSum += (bigList[y][x])
# averagePoints.append(float(pointsToSum/MAX_ITERATIONS))
  

plotGraph()