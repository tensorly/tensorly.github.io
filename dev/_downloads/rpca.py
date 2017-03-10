# -*- coding: utf-8 -*-
"""
Robust Tensor Robust PCA
========================

Example on how to use :mod:`tensorly.decomposition.robust_pca` to perform Robust Tensor PCA.

"""
import matplotlib.pyplot as plt
from tensorly.datasets.yaleb import fetch_cropped_yaleb

dataset_path = '/data/tensorly_data/'
data = fetch_cropped_yaleb(dataset_path, zooming=0.3, max_n_subjects=5)

###########################################################################
# Accumulate a tensor containing all the data
X = np.concatenate([data[key]['images'] for key in data], axis=0)
print(X.shape)

###########################################################################
# Convert to float
X = X.astype(np.float64)
X -= X.mean()

###########################################################################
# Visualise the data

def visualise_images(X, n_images, n_columns, randomise=True):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    indices = indices[:n_images]
    cmap = plt.cm.Greys_r
    n_rows = np.ceil(n_images / n_columns)
    fig = plt.figure(figsize=(2*n_columns, 2*n_rows))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i, e in enumerate(indices):
        ax = fig.add_subplot(n_rows, n_columns, i + 1, xticks=[], yticks=[])
        ax.imshow(X[e], cmap=cmap, interpolation='nearest')

visualise_images(X, 12, 4)
plt.show()

###########################################################################
# Add noise
from tensorly.random.noise import add_noise

X = add_noise(X, noise='salt_pepper', percent=0.15, inplace=True, random_state=random_state)
visualise_images(X, 12, 4)
plt.show()

###########################################################################
# Apply robust pca
from tensorly.decomposition import robust_pca
low_rank_part, sparse_part = robust_pca(X, reg_E=0.04, learning_rate=1.2, n_iter_max=20)


###########################################################################
# Check the results
def visualise_rpca(X, low_rank_part, sparse_part, n_images=10):
    """A little helper function to visualise the result of tensor RPCA
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    indices = indices[:n_images]

    fig = plt.figure(figsize=(10, 2*n_images))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i, e in enumerate(indices):
        cmap = plt.cm.Greys_r

        ax = fig.add_subplot(n_images, 4, 4*i + 1, xticks=[], yticks=[])
        ax.imshow(X[e], cmap=cmap, interpolation='nearest')
        if not i:
            ax.set_title('Original')

        ax = fig.add_subplot(n_images, 4, 4*i + 2, xticks=[], yticks=[])
        ax.imshow(low_rank_part[e], cmap=cmap, interpolation='nearest')
        if not i:
            ax.set_title('Low-rank')

        ax = fig.add_subplot(n_images, 4, 4*i + 3, xticks=[], yticks=[])
        ax.imshow(sparse_part[e], cmap=cmap, interpolation='nearest')
        if not i:
            ax.set_title('Sparse')

        ax = fig.add_subplot(n_images, 4, 4*i + 4, xticks=[], yticks=[])
        ax.imshow(low_rank_part[e]+sparse_part[e], cmap=cmap, interpolation='nearest')
        if not i:
            ax.set_title('Reconstruction')

    return fig

visualise_rpca(X, low_rank_part, sparse_part, 6)
plt.show()
