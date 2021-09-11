import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

# Something like...
#numpy.random.uniform(0.0, 1.0, (1000))
#numpy.random.uniform(0.0, 1.0, (1000,2))
#numpy.random.uniform(0.0, 1.0, (1000,3,3))
#numpy.random.uniform(0.0, 1.0, (1000,4,4,4))
#numpy.random.uniform(0.0, 1.0, (1000,5,5,5,5))
# Nope lol nvm.. np.random.rand(N,d)

## 1000 points under one dimension
points = np.random.rand(10,3)
print(points)

def findMinMax(p):
  #d = np.diff(p, axis=0)
  #print(d)
  #segmentDistances = np.hypot(d[:,0], d[:,1]) #Wrong
  #print(max(segmentDistances)) #Wrong
  #print(min(segmentDistances))

  hull = ConvexHull(points)
  # Extract the points forming the hull
  hullpoints = points[hull.vertices,:]
  # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
  hdist = cdist(hullpoints, hullpoints, metric='euclidean')
  # Get the farthest apart points
  maxpair = np.unravel_index(hdist.argmax(), hdist.shape)
  minpair = np.unravel_index(hdist.argmin(), hdist.shape)

  print(hullpoints[maxpair[0]], hullpoints[maxpair[1]])
  print(hullpoints[minpair[0]], hullpoints[minpair[1]])


  #print([hullpoints[bestpair[0]],hullpoints[bestpair[1]]])
  #print(np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]]))

  pairPoints = np.array([(np.linalg.norm(hullpoints[minpair[0]] - hullpoints[minpair[1]])),(np.linalg.norm(hullpoints[maxpair[0]] - hullpoints[maxpair[1]]))])
  print(pairPoints)

  logDiff = np.log10((pairPoints[1]-pairPoints[0])/pairPoints[0])
  print(logDiff)
  #listOfLogDiffs.append(logDiff)

findMinMax(points)

listOfLogDiffs = []

#for x in range(2,3):
#  findMinMax(np.random.rand(1000,x))
#print(listOfLogDiffs)

## Plotting the log results
def plotGraph():
  x = np.arange(2,51)
  y = listOfLogDiffs
  plt.title("Curse of Dimensionality") 
  plt.xlabel("Number of Dimensions") 
  plt.ylabel("Log_10 * ((Max-Min)/Min)") 
  plt.plot(x,y) 
  plt.show()