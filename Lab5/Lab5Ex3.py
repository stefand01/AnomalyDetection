import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = scipy.io.loadmat('shuttle.mat')  
X = data['X'] 
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()  
X_train_normalized = scaler.fit_transform(X_train) 
X_test_normalized = scaler.transform(X_test)  

print(f"Shape of X_train: {X_train_normalized.shape}")
print(f"Shape of X_test: {X_test_normalized.shape}")
print(f"Sample normalized X_train:\n{X_train_normalized[:5]}")