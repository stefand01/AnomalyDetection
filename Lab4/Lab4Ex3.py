import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.preprocessing import MinMaxScaler

data = scipy.io.loadmat('shuttle.mat')  

X = data['X']
y = data['y'].ravel()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
clf_name = 'OneClassSVM'
clf = OCSVM(kernel = 'rbf')
clf.fit(x_train)
y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_ 

y_test_pred = clf.predict(x_test)
y_test_scores = clf.decision_function(x_test)

balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
rocauc_score = roc_auc_score(y_test,y_test_scores)
print(balanced_accuracy)
print(rocauc_score)

# clf2_name = 'DeepSVDD'
# clf2_architecture = ''
# clf2 = DeepSVDD(n_features = 2, hidden_activation = 'gelu', use_ae = True)
# clf2.fit(x_train)
# y_train_pred2 = clf2.labels_
# y_train_scores2 = clf2.decision_scores_ 

# y_test_pred2 = clf2.predict(x_test)
# y_test_scores2 = clf2.decision_function(x_test)

# balanced_accuracy2 = balanced_accuracy_score(y_test, y_test_pred2)
# rocauc_score2 = roc_auc_score(y_test,y_test_scores2)
# print(balanced_accuracy2)
# print(rocauc_score2)