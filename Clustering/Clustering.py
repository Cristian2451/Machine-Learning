# Using K-means and DBSCAN to cluster a set of data with two features (x1 and x2).
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

points = loadtxt('K_means_test_set.csv', delimiter=',')
plt.style.use("dark_background")
plt.plot(points[:,0],points[:,1],'b.')
plt.show()

## K-means Clustering
model = KMeans(n_clusters = 5, n_init = 10, max_iter = 300).fit(points)

cs = ['blue', 'red', 'green', 'tab:orange','tab:purple']
labels_to_colors = [cs[int(lbl)] for lbl in model.labels_]

plt.scatter(points[:, 0], points[:, 1], c = labels_to_colors, alpha=0.5)
plt.plot(model.cluster_centers_[:,0],model.cluster_centers_[:,1],'wx', mew=3, ms=6);
plt.show()

# Perform Elbow Method by plotting the average distance of points from centroids as a function of no. of clusters.
# In this case, 3 clusters are optimal.
Inertia_k = []
for k in range(1,10):
    model = KMeans(n_clusters = k, n_init = 10, max_iter = 300).fit(points)
    Inertia_k.append(model.inertia_)
plt.semilogy(range(1,10),Inertia_k);
p = plt.plot(3,Inertia_k[2],'rx',ms=6)
plt.legend(p,{'inflection point'})
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

model = KMeans(n_clusters = 3, n_init = 10, max_iter = 300).fit(points)
labels_to_colors = [cs[int(lbl)] for lbl in model.labels_]
plt.scatter(points[:, 0], points[:, 1], c=labels_to_colors, alpha=0.5)
plt.plot(model.cluster_centers_[:,0],model.cluster_centers_[:,1],'wx', mew=3, ms=6)

# Predict labels for new points pt0, pt1, pt2
pt0 = np.array([[-2,-2]])
pt1 = np.array([[ 4, 0]])
pt2 = np.array([[-2, 2]])

plt.plot(pt0[:, 0], pt0[:, 1], 'o', markeredgecolor='pink', mew=3, ms=8, color='w')
plt.plot(pt1[:, 0], pt1[:, 1], 'o', markeredgecolor='yellow', mew=3, ms=8, color='w')
plt.plot(pt2[:, 0], pt2[:, 1], 'o',  markeredgecolor='cyan', mew=3, ms=8, color='w')
plt.show()

print('pink point class: ', model.predict(pt0)[0],', class color ', cs[model.predict(pt0)[0]])
print('yellow point class: ', model.predict(pt1)[0],', class color ', cs[model.predict(pt1)[0]])
print('cyan point class: ', model.predict(pt2)[0],', class color ', cs[model.predict(pt2)[0]])

## DBSCAN
# Train model specifying:
# - maximum distance between two samples (eps)
# - The number of samples in a neighborhood for a point to be considered as a core point.
DB_model = DBSCAN(eps = 0.3, min_samples = 10).fit(points)
labels_to_colors = [cs[int(lbl)] for lbl in DB_model.labels_]
plt.scatter(points[:, 0], points[:, 1], c = labels_to_colors, alpha=0.5)
plt.show()

axs[1].axis("off")
axs[1].set_title(f"Quantized image ({n_colors} colors, K-Means)", fontsize=24);
axs[1].imshow(img_rec);
plt.show()
