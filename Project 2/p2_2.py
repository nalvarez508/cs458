import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, tree, metrics

# Generate dataset
## 5000 instances (Gaussian)
gaus_center = np.random.normal(loc=np.array([10,10]), scale=np.sqrt(2), size=(5000,2))
### 200 instances (Uniform)
gaus_noise = np.random.uniform(low=0, high=20, size=(200,2))
c1 = np.concatenate((gaus_center, gaus_noise), axis=0)

## 5200 instances (Uniform)
c2 = np.random.uniform(low=0, high=20, size=(5200,2))

plt.scatter(c2[:, 0], c2[:, 1], c='red', marker='.', s=2.5)
plt.scatter(c1[:, 0], c1[:, 1], c='blue', marker='+', s=2.5)

fig, axs = plt.subplots(1,2)

c3 = np.concatenate((c1,c2), axis=0)
c3_target = np.concatenate((np.zeros((c1.shape[0],1)), np.ones((c2.shape[0],1))), axis=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(c3, c3_target, test_size=0.1, random_state=0, shuffle=True)

TrainError = np.empty((0,2))
TestError = np.empty((0,2))
for nodes in range(2, 151):
  clf = tree.DecisionTreeClassifier(max_leaf_nodes=nodes)
  clf.fit(X_train, y_train)
  y_pred_train = clf.predict(X_train)
  y_pred_test = clf.predict(X_test)
  TrainError = np.append(TrainError, np.array([(nodes, 1-metrics.accuracy_score(y_train, y_pred_train))]), axis=0)
  TestError = np.append(TestError, np.array([(nodes, 1-metrics.accuracy_score(y_test, y_pred_test))]), axis=0)

axs[0].plot(TrainError[:9, 0], TrainError[:9, 1], c='blue', marker='o', markersize=2)
axs[0].plot(TestError[:9, 0], TestError[:9, 1], c='red', marker='o', markersize=2)
axs[1].plot(TrainError[:, 0], TrainError[:, 1], c='blue', marker='o', markersize=2)
axs[1].plot(TestError[:, 0], TestError[:, 1], c='red', marker='o', markersize=2)

axs[1].set_xlabel("Number of nodes")
axs[0].set_ylabel("Error rate")
plt.show()
