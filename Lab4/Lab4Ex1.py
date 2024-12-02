from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

def plot_3d(X, y, title, ax):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='blue', label='Inliers')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='red', label='Outliers')
    ax.set_title(title)
    ax.legend()

n_train = 300
n_test = 200
features = 3
contamination = 0.15

x_train, x_test, y_train, y_test = generate_data(n_train=n_train, n_test = n_test, n_features = features, contamination = contamination)

clf_name = 'OneClassSVM'
clf = OCSVM(contamination=contamination, kernel = 'rbf')
clf.fit(x_train)
y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_ 

y_test_pred = clf.predict(x_test)
y_test_scores = clf.decision_function(x_test)

balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
rocauc_score = roc_auc_score(y_test,y_test_scores)
print(balanced_accuracy)
print(rocauc_score)

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(221, projection='3d')
plot_3d(x_train, y_train, "Ground Truth (Train Data)", ax1)
ax2 = fig.add_subplot(222, projection='3d')
plot_3d(x_train, y_train_pred, "OCSVM Predictions Train", ax2)
ax3 = fig.add_subplot(223, projection='3d')
plot_3d(x_test, y_test, "Ground Truth (Test Data)", ax3)
ax4 = fig.add_subplot(224, projection='3d')
plot_3d(x_test, y_test_pred, "OCSVM Predictions Test", ax4)
plt.show()


clf2_name = 'DeepSVDD'
clf2 = DeepSVDD(contamination=contamination, n_features = 3)
clf2.fit(x_train)
y_train_pred2 = clf2.labels_
y_train_scores2 = clf2.decision_scores_ 

y_test_pred2 = clf2.predict(x_test)
y_test_scores2 = clf2.decision_function(x_test)

balanced_accuracy2 = balanced_accuracy_score(y_test, y_test_pred2)
rocauc_score2 = roc_auc_score(y_test,y_test_scores2)
print(balanced_accuracy2)
print(rocauc_score2)

fig2 = plt.figure(figsize=(16, 12))
ax1 = fig2.add_subplot(221, projection='3d')
plot_3d(x_train, y_train, "Ground Truth (Train Data)", ax1)
ax2 = fig2.add_subplot(222, projection='3d')
plot_3d(x_train, y_train_pred2, "Deep SVDD Predictions Train", ax2)
ax3 = fig2.add_subplot(223, projection='3d')
plot_3d(x_test, y_test, "Ground Truth (Test Data)", ax3)
ax4 = fig2.add_subplot(224, projection='3d')
plot_3d(x_test, y_test_pred2, "Deep SVDD Predictions Test", ax4)
plt.show()


