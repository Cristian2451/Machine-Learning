import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt

points = loadtxt('K_means_test_set.csv', delimiter=',')
plt.plot(points[:,0],points[:,1],'b.')
plt.show()


from sklearn.cluster import KMeans
model = KMeans(n_clusters = 5, n_init = 10, max_iter = 300).fit(points)

print(model.labels_) #shows all labels assigned to the training points
print(model.cluster_centers_) #gives coordinates to all final centroids

cs = ['blue', 'yellow', 'green', 'tab:orange','tab:purple']
labels_to_colors = [cs[int(lbl)] for lbl in model.labels_]

plt.scatter(points[:, 0], points[:, 1], c = labels_to_colors, alpha=0.5)
plt.plot(model.cluster_centers_[:,0],model.cluster_centers_[:,1],'kx', mew=3, ms=6);
plt.show()

Inertia_k = []
for k in range(1,10):
    model = KMeans(n_clusters = k, n_init = 10, max_iter = 300).fit(points)
    Inertia_k.append(model.inertia_)
plt.semilogy(range(1,10),Inertia_k);
plt.show()

model = KMeans(n_clusters = 3, n_init = 10, max_iter = 300).fit(points)
labels_to_colors = [cs[int(lbl)] for lbl in model.labels_]

plt.scatter(points[:, 0], points[:, 1], c=labels_to_colors, alpha=0.5)
plt.plot(model.cluster_centers_[:,0],model.cluster_centers_[:,1],'kx', mew=3, ms=6)

pt0 = np.array([[-2,-2]])
pt1 = np.array([[ 4, 0]])
pt2 = np.array([[-2, 2]])

plt.plot(pt0[:, 0], pt0[:, 1], 'o', markeredgecolor='tab:pink', mew=3, ms=8, color='w')
plt.plot(pt1[:, 0], pt1[:, 1], 'o', markeredgecolor='tab:red', mew=3, ms=8, color='w')
plt.plot(pt2[:, 0], pt2[:, 1], 'o',  markeredgecolor='tab:cyan', mew=3, ms=8, color='w')
plt.show()

print('pink point class: ', model.predict(pt0)[0],', class color ', cs[model.predict(pt0)[0]])
print('red point class: ', model.predict(pt1)[0],', class color ', cs[model.predict(pt1)[0]])
print('cyan point class: ', model.predict(pt2)[0],', class color ', cs[model.predict(pt2)[0]])

pt3 = np.array([[0, 0]])
print('pt3 class: ', model.predict(pt3)[0],', class color ', cs[model.predict(pt3)[0]])

# DBSCAN

from sklearn.cluster import DBSCAN
DB_model = DBSCAN(eps = 0.3, min_samples = 10).fit(points)

labels_to_colors = [cs[int(lbl)] for lbl in DB_model.labels_]
plt.scatter(points[:, 0], points[:, 1], c = labels_to_colors, alpha=0.5)
plt.show()

# Color quantization using K-means

# Load photo
img_original = plt.imread("Sommaroy.jpg")
plt.figure(figsize = (10,10))
plt.imshow(img_original)
plt.axis('off');

# Number of unique colors:
num_colors = np.unique(img_original.reshape(-1, img_original.shape[2]), axis = 0)
plt.title(f"Original image ({len(num_colors)} colors)");
plt.show()


from sklearn.utils import shuffle
# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = np.array(img_original, dtype = np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img_original.shape)
image_array = np.reshape(img, (w * h, d))

n_colors = 32
image_array_sample = shuffle(image_array, random_state = 0, n_samples = 1000)
kmeans = KMeans(n_clusters = n_colors, random_state = 0, n_init=10).fit(image_array_sample)

# Get labels for all points
labels = kmeans.predict(image_array)

# Recreate Images:
img_rec = kmeans.cluster_centers_[labels].reshape(w, h, -1)

# img_rec = recolor_image(n_colors=32) # Try other colours, e.g., 32, 1000, 10, 5


# Display all results, alongside original image
f, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(24, 12), layout='tight');
axs[0].axis('off');
axs[0].set_title(f"Original image ({len(num_colors)} colors)", fontsize=24);
axs[0].imshow(img)

axs[1].axis("off")
axs[1].set_title(f"Quantized image ({n_colors} colors, K-Means)", fontsize=24);
axs[1].imshow(img_rec);
plt.show()