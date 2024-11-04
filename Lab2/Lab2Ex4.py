import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.models.combination import maximization,average

data = scipy.io.loadmat('cardio.mat')  
x = data['X'] 
y = data['y'].ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
                                                    
x_train = standardizer(x_train)
x_test = standardizer(x_test)

n_neighbors_range = range(30, 121, 10)
contamination = 0.05       

train_scores_list = []
test_scores_list = []

for n_neighbors in n_neighbors_range:
    knn = KNN(n_neighbors = n_neighbors, contamination = contamination)
    knn.fit(x_train)

    train_scores = knn.decision_scores_
    test_scores = knn.decision_function(x_test)
    train_scores_list.append(train_scores)
    test_scores_list.append(test_scores)
    y_train_pred = knn.labels_
    y_test_pred = knn.predict(x_test)
    
    print("=====================================")
    print(f"n_neighbors={n_neighbors}")
    print(">>>><<<<")
    print("Train:")
    print("Balanced Accuracy:")
    print(balanced_accuracy_score(y_train,y_train_pred))
    print("------")
    print("Train:")
    print("Balanced Accuracy:")
    print(balanced_accuracy_score(y_train,y_train_pred))
print("=====================================")

train_scores_list = standardizer(train_scores_list)
test_scores_list = standardizer(test_scores_list)
avg_train_scores = average(train_scores_list)
avg_test_scores = average(test_scores_list)
max_train_scores = maximization(train_scores_list)
max_test_scores = maximization(test_scores_list)
threshold_avg = np.quantile(avg_train_scores, 1 - contamination)
threshold_max = np.quantile(max_train_scores, 1 - contamination)
y_train_pred_avg = (avg_train_scores > threshold_avg).astype(int)
y_test_pred_avg = (avg_test_scores > threshold_avg).astype(int)
y_train_pred_max = (max_train_scores > threshold_max).astype(int)
y_test_pred_max = (max_test_scores > threshold_max).astype(int)
print("Balanced Accuracy (Train):", balanced_accuracy_score(y_train, y_train_pred_avg))
print("Balanced Accuracy (Test):", balanced_accuracy_score(y_test, y_test_pred_avg))
print("Balanced Accuracy (Train):", balanced_accuracy_score(y_train, y_train_pred_max))
print("Balanced Accuracy (Test):", balanced_accuracy_score(y_test, y_test_pred_max))
