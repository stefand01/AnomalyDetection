from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pyod.models.lof import LOF
from pyod.models.knn import KNN
import numpy as np
n_blob_a = 200
n_blob_b = 100
n_neighbours = [3 , 5 , 11 , 21]
contamination = 0.07
x_blob_a, y_blob_a = make_blobs(n_samples=n_blob_a,n_features=2,center_box=(-10,-10),cluster_std=2)
x_blob_b, y_blob_b = make_blobs(n_samples=n_blob_b,n_features=2,center_box=(10,10),cluster_std=6)

x_data = np.vstack((x_blob_a, x_blob_b))
for n in n_neighbours:
    lof = LOF(n_neighbors=n, contamination=contamination)
    knn = KNN(n_neighbors=n, contamination=contamination)
    lof.fit(x_data)
    knn.fit(x_data)
    y_pred_lof = lof.labels_
    y_pred_knn = knn.labels_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(x_data[:, 0], x_data[:, 1], c=y_pred_knn, cmap='coolwarm', s=15)
    ax1.set_title("KNN Detection")
    ax1.set_title(f"LOF Detection (n_neighbors={n})")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax2.scatter(x_data[:, 0], x_data[:, 1], c=y_pred_lof, cmap='coolwarm', s=15)
    ax2.set_title("LOF Detection")
    ax2.set_title(f"LOF Detection (n_neighbors={n})")
    ax2.set_xlabel("Feature 1")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()