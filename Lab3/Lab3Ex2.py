import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest

centers = [(10, 0), (0, 10)]
cluster_std = 1
n_samples = 500

X_train, _ = make_blobs(n_samples=n_samples*2, centers=centers, cluster_std=cluster_std, random_state=42)

contamination = 0.02
iforest = IForest(contamination=contamination, random_state=42)
iforest.fit(X_train)


test_data = np.random.uniform(-10, 20, size=(1000, 2))


anomaly_scores = iforest.decision_function(test_data) 

plt.figure(figsize=(10, 8))
scatter = plt.scatter(test_data[:, 0], test_data[:, 1], c=anomaly_scores, s=10)
plt.colorbar(scatter, label='Anomaly Score')
plt.title('Test Data Anomaly Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
