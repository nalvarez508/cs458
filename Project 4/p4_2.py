from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# (a) generate swiss roll dataset
n_samples = 1500
noise = 0.05
x, _ = datasets.make_swiss_roll(n_samples, noise=noise)
x[:, 1] *= .5

# (b) Agglomerative Clustering
