import sklearn
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.example import visualize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
n_train = 400
n_test = 100
contamination = 0.1
x_train, x_test, y_train, y_test = generate_data(n_train = n_train, n_test = n_test, 
                                                 n_features = 2, contamination = contamination)


plt.figure(figsize=(8, 6))
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], c='blue', label='Normal')
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], c='red', label='Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


clf_name = 'KNN'
clf = KNN(contamination=0.2)
clf.fit(x_train, y_train)

y_train_pred = clf.labels_
y_test_pred = clf.predict(x_test)

print(confusion_matrix(y_test,y_test_pred))
print(balanced_accuracy_score(y_test,y_test_pred))

y_test_scores = clf.decision_function(x_test)

fpr, tpr, tresholds = roc_curve(y_test, y_test_scores) #iau scorurile
roc_auc = auc(fpr, tpr)  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.scatter(x, y)
plt.plot(fpr, fpr)
plt.show()
