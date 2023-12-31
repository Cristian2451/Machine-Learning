# Color quantization using K-means
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load photo
img_original = plt.imread("Sommaroy.jpg")
plt.figure(figsize = (10,10))
plt.imshow(img_original)
plt.axis('off')

# Number of unique colors:
num_colors = np.unique(img_original.reshape(-1, img_original.shape[2]), axis = 0)
plt.title(f"Original image ({len(num_colors)} colors)")
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

# img_rec = recolor_image(n_colors=32) # Try other colours

# Display all results, alongside original image
f, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(24, 12), layout='tight')
axs[0].axis('off')
axs[0].set_title(f"Original image ({len(num_colors)} colors)", fontsize=24)
axs[0].imshow(img)

axs[1].axis("off")
axs[1].set_title(f"Quantized image ({n_colors} colors, K-Means)", fontsize=24)
axs[1].imshow(img_rec)
plt.show()