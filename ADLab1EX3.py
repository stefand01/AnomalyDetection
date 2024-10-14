import sklearn
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.example import visualize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
n_train = 1000
n_test = 0
contamination = 0.1
x_train, x_test, y_train, y_test = generate_data(n_train = n_train, 
                                                 n_features = 1, contamination = contamination)

x_train = x_train.flatten()

mean = np.mean(x_train)
std = np.std(x_train)
z_scores = (x_train-mean)/std

threshold = np.quantile(np.abs(z_scores),0.9)
y_train_pred = (np.abs(z_scores) > threshold).astype(int)

balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

print(mean)
print(std)
print(balanced_acc)