import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from progressbar import progressbar

MAX_ITERATIONS = 15
NUMBER_OF_POINTS = 100

## 1000 points under one dimension
#points = np.random.rand(10,2)
#print(points)

def findMinMax(p):
  #d = np.diff(p, axis=0)
  #print(d)
  #segmentDistances = np.hypot(d[:,0], d[:,1]) #Wrong
  #print(max(segmentDistances)) #Wrong
  #print(min(segmentDistances))

  hull = ConvexHull(p)
  # Extract the points forming the hull
  hullpoints = p[hull.vertices,:]
  # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
  hdist = cdist(hullpoints, hullpoints, metric='euclidean')
  # Get the farthest apart points
  maxpair = np.unravel_index(hdist.argmax(), hdist.shape)
  i,j = np.where(hdist==np.min(hdist[np.nonzero(hdist)]))

  #print(hullpoints[maxpair[0]], hullpoints[maxpair[1]])
  #print(hullpoints[i][0], hullpoints[i][1])

  pairPoints = np.array([(np.linalg.norm(hullpoints[i][0] - hullpoints[i][1])),(np.linalg.norm(hullpoints[maxpair[0]] - hullpoints[maxpair[1]]))])
  #print(pairPoints)

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

for x in progressbar(range(2,MAX_ITERATIONS), redirect_stdout=True):
#for x in range(2,MAX_ITERATIONS):
  print(f"({x}/{MAX_ITERATIONS}): ", end='')
  findMinMax(np.random.rand(NUMBER_OF_POINTS,x))
#print(listOfLogDiffs)

plotGraph()