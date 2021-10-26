import numpy as np
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt

# (a) generate 2-class data points
np.random.seed(0)
x = np.random.rand(300,2) * 10 - 5
y = np.logical_xor(x[:,0]>0, x[:,1]>0)

# (b) develop nonlinear SVM binary classifier
clf = NuSVC(gamma='auto')
clf.fit(x,y)

# (c) plot the decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)

#plt.imshow(z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower')
cont = plt.contour(xx, yy, z, levels=[0], linewidths=2)
plt.scatter(x[:,0], x[:,1], s=30, c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.show()