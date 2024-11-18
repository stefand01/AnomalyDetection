import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF

data = loadmat('shuttle.mat')  
X = data['X']
y = data['y'].ravel()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

models = {
    'IForest': IForest(contamination=0.1),
    'LODA': LODA(contamination=0.1),
    #'DIF': DIF(contamination=0.1) se incarca foarte greu
}

results = {model_name: {'BA': [], 'ROC_AUC': []} for model_name in models.keys()}

for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None)
    
    for model_name, model in models.items():
        model.fit(X_train)
        
        y_scores = model.decision_function(X_test)  
        y_pred = model.predict(X_test)  
       
        ba = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        
        results[model_name]['BA'].append(ba)
        results[model_name]['ROC_AUC'].append(roc_auc)

for model_name in results.keys():
    mean_ba = np.mean(results[model_name]['BA'])
    mean_roc_auc = np.mean(results[model_name]['ROC_AUC'])
    print(f"{model_name}: Mean Balanced Accuracy = {mean_ba:.4f}, Mean ROC AUC = {mean_roc_auc:.4f}")
