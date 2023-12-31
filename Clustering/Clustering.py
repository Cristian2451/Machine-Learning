# Using K-means and DBSCAN to cluster a set of data with two features (x1 and x2).
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

points = loadtxt('data.csv', delimiter=',')
plt.style.use("dark_background")
plt.plot(points[:,0],points[:,1],'b.',ms=2)
plt.show()

## K-means Clustering
model = KMeans(n_clusters = 5, n_init = 10, max_iter = 300).fit(points)

cs = ['blue', 'yellow', 'green','red','pink','cyan','magenta','orange','purple','white','lime','gold','grey', 'tab:orange','tab:green','tab:purple']
labels_to_colors = [cs[int(lbl)] for lbl in model.labels_]

plt.scatter(points[:, 0], points[:, 1], c = labels_to_colors, alpha=0.5, s=2)
plt.plot(model.cluster_centers_[:,0],model.cluster_centers_[:,1],'wx', ms=4);
plt.show()

# Perform Elbow Method by plotting the average distance of points from centroids as a function of no. of clusters.
# In this case, 3 clusters are optimal.
Inertia_k = []
for k in range(1,30):
    model = KMeans(n_clusters = k, n_init = 10, max_iter = 300).fit(points)
    Inertia_k.append(model.inertia_)
plt.semilogy(range(1,30),Inertia_k);
p = plt.plot(15,Inertia_k[14],'rx',ms=6)
plt.legend(p,{'inflection point'})
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

model = KMeans(n_clusters = 15, n_init = 10, max_iter = 300).fit(points)
labels_to_colors = [cs[int(lbl)] for lbl in model.labels_]
plt.scatter(points[:, 0], points[:, 1], c=labels_to_colors, alpha=0.5,s=2)
plt.plot(model.cluster_centers_[:,0],model.cluster_centers_[:,1],'wx', ms=4)
plt.show()

# Predict labels for new points pt0, pt1, pt2
pt0 = np.array([[8e5,4e5]])
pt1 = np.array([[3e5, 3.5e5]])
pt2 = np.array([[1.5e5, 9e5]])

print('pt0: ', model.predict(pt0)[0],', class color ', cs[model.predict(pt0)[0]])
print('pt1: ', model.predict(pt1)[0],', class color ', cs[model.predict(pt1)[0]])
print('pt2: ', model.predict(pt2)[0],', class color ', cs[model.predict(pt2)[0]])

## DBSCAN
# Train model specifying:
# - maximum distance between two samples (eps)
# - The number of samples in a neighborhood for a point to be considered as a core point.
DB_model = DBSCAN(eps = 20000, min_samples = 15).fit(points)
labels_to_colors = [cs[int(lbl)] for lbl in DB_model.labels_]
plt.scatter(points[:, 0], points[:, 1], c = labels_to_colors, alpha=0.5,s=2)
plt.show()
