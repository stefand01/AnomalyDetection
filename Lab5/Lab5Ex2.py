import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
data = scipy.io.loadmat('shuttle.mat')

x = data['X']
y = data['y'].ravel()
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)
clf_name = 'PCA'
clf2_name = 'KPCA'
clf = PCA()
clf2 = KPCA()
clf.fit(x_train)
explained_variances = clf.explained_variance_ratio_
cumulative_variances = np.cumsum(explained_variances)

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variances) + 1), explained_variances, alpha=0.6, label='Individual Variance')
plt.step(range(1, len(cumulative_variances) + 1), cumulative_variances, where='mid', label='Cumulative Variance', color='red')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance (PCA)')
plt.legend()
plt.grid(True)
plt.show()

clf2.fit(x_train)
y_train_pred = clf.labels_
y_train_pred2 = clf2.labels_
y_train_scores = clf.decision_scores_
y_train_scores2 = clf2.decision_scores_

y_test_pred = clf.predict(x_test)
y_test_pred2 = clf2.predict(x_test)
y_test_scores = clf.decision_function(x_test)
y_test_scores2 = clf2.decision_function(x_test)

balanced_accuracy_test = balanced_accuracy_score(y_test, y_test_pred)
balanced_accuracy_train = balanced_accuracy_score(y_train, y_train_pred)
balanced_accuracy2_test = balanced_accuracy_score(y_test, y_test_pred2)
balanced_accuracy2_train = balanced_accuracy_score(y_train, y_train_pred2)
print(clf_name)
print("Balanced acc train: ", balanced_accuracy_train)
print("Balanced acc test: ", balanced_accuracy_test)
print(clf2_name)
print("Balanced acc train: ", balanced_accuracy2_train)
print("Balanced acc test: ", balanced_accuracy2_test)


