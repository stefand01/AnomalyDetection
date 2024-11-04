from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

n_train = 400
n_test = 200
n_clusters = 2
contamination = 0.1
n_features = 2
x_train , x_test , y_train , y_test = generate_data_clusters(n_test=n_test, n_clusters=n_clusters, n_train=n_train, n_features=n_features,contamination=contamination)
n_neighbours = [3 , 5 , 11 , 21]
for n in n_neighbours:
    knn = KNN(n_neighbors=n, contamination=contamination)
    knn.fit(x_train)
    y_train_pred = knn.labels_
    y_train_scores = knn.decision_scores_
    y_test_pred = knn.predict(x_test)
    y_test_scores = knn.decision_function(x_test)

    print("Train:")
    evaluate_print(n, y_train, y_train_scores)
    print("Balanced Accuracy:")
    print(balanced_accuracy_score(y_train,y_train_pred))

    print("Test:")
    evaluate_print(n, y_test, y_test_scores)
    print("Balanced Accuracy:")
    print(balanced_accuracy_score(y_test,y_test_pred))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"KNN with n_neighbors={n}")
    axs[0, 0].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', s=15)
    axs[0, 0].set_title("Ground Truth (Train)")
    axs[0, 0].set_xlabel("Feature 1")
    axs[0, 0].set_ylabel("Feature 2")
    axs[0, 1].scatter(x_train[:, 0], x_train[:, 1], c=y_train_pred, cmap='coolwarm', s=15)
    axs[0, 1].set_title("Predicted Labels (Train)")
    axs[0, 1].set_xlabel("Feature 1")
    axs[0, 1].set_ylabel("Feature 2")
    axs[1, 0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm', s=15)
    axs[1, 0].set_title("Ground Truth (Test)")
    axs[1, 0].set_xlabel("Feature 1")
    axs[1, 0].set_ylabel("Feature 2")
    axs[1, 1].scatter(x_test[:, 0], x_test[:, 1], c=y_test_pred, cmap='coolwarm', s=15)
    axs[1, 1].set_title("Predicted Labels (Test)")
    axs[1, 1].set_xlabel("Feature 1")
    axs[1, 1].set_ylabel("Feature 2")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()