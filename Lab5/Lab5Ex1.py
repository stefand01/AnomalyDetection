import numpy as np
import matplotlib.pyplot as plt

data = np.random.multivariate_normal(mean = [5,10,2], cov = [[3,2,2], [2,10,1], [2,1,2]], size = 500)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(data[:,0], data[:,1], data[:,2], alpha = 0.7, label = "Data points")
ax.set_title("3D Dataset")
plt.legend()
plt.show()

mean_data = np.mean(data, axis=0)
centered_data = data - mean_data
cov_matrix = np.cov(centered_data, rowvar = False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
A = np.diag(eigenvalues)
INV = np.linalg.inv(eigenvectors)
Decomposed = np.dot(np.dot(eigenvectors,A), INV)
print("EVD:", Decomposed)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print("Descending eigenvalues", eigenvalues)
print("Descending eigenvectors", eigenvectors)

cum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
ind_variance = eigenvalues / np.sum(eigenvalues)

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(ind_variance)+1), ind_variance, alpha=0.6, label='Individual Variance')
plt.step(range(1, len(cum_variance)+1), cum_variance, where='mid', label='Cumulative Variance', color='red')
plt.legend()
plt.grid(True)
plt.show()