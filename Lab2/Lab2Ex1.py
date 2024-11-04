import pyod
from sklearn.linear_model import LinearRegression
import numpy as np

def leverage_scores(X):
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    leverage_scores = np.diag(H)
    return leverage_scores

n_points = 100
mu_values = [0 , 1]
sigma_values = [0.5 , 2]

x_regular = np.random.normal(0,1,n_points)
y_regular = np.random.normal(0,1,n_points)


x_highvar_x = np.random.normal(mu, sigma * sigma, n_points)
y_highvar_x = np.random.normal(0,1,n_points)

x_highvar_y = np.random.normal(0,1,n_points)
y_highvar_y = np.random.normal(mu, sigma * sigma, n_points)

x_highvar_xy = np.random.normal(mu, sigma * sigma, n_points)
y_highvar_xy = np.random.normal(mu, sigma * sigma, n_points)
print(x_regular)

