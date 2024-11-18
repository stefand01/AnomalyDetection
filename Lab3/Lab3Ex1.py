import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1, random_state=42)

num_projections = 5
projections = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=num_projections)
projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

num_bins = 10   
histograms = []
bin_edges = []


for proj in projections:
    projected_data = X @ proj  
    hist, edges = np.histogram(projected_data, bins=num_bins, range=(-6, 6), density=True)
    histograms.append(hist)
    bin_edges.append(edges)

X_test = np.random.uniform(-3, 3, size=(500, 2))


anomaly_scores = []
for point in X_test:
    scores = []
    for proj, hist, edges in zip(projections, histograms, bin_edges):
        projected_value = point @ proj
        bin_idx = np.digitize(projected_value, edges) - 1
        if bin_idx < 0 or bin_idx >= len(hist):
            scores.append(0) 
        else:
            scores.append(hist[bin_idx])
    anomaly_scores.append(np.mean(scores))

anomaly_scores = np.array(anomaly_scores)

plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_scores, s=30)
plt.colorbar(label="Anomaly Score")
plt.title("Anomaly Score Map (Number of Bins = 10)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
