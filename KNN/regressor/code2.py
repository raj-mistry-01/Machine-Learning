# without library
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 

data = pd.read_csv(r"csvs\BostonHousing.csv", header='infer').values
x = data[:,:-1]  
y = data[:, -1]  

test_split = float(input("Enter a number between 0 to 1 to specify how much data is required as the test data: "))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
k = int(input("Enter the number of nearest neighbours to be used, i.e. k: "))
dist = np.zeros(shape=x_train.shape[0])
pred = np.zeros(shape=x_test.shape[0])
for i in range(x_test.shape[0]): 
    dist = np.sqrt(np.sum((x_train - x_test[i]) ** 2, axis=1)) 
    KMinimumDistance = np.argpartition(dist, k)[:k] 
    inv_dist = 1 / (dist[KMinimumDistance] + 1e-20) 
    denom = np.sum(inv_dist)  
    pred[i] = np.dot(inv_dist / denom, y_train[KMinimumDistance])  

def MAE(pred, y_test):
    return np.mean(abs(pred - y_test))

def MSE(pred, y_test):
    return np.mean((pred - y_test) ** 2)

def MAPE(pred, y_test):
    return np.mean(abs((pred - y_test) / y_test))

mae = MAE(pred, y_test)
mse = MSE(pred, y_test)
rmse = np.sqrt(mse)
mape = MAPE(pred, y_test)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
