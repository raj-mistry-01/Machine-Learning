import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"csvs\Iris.csv",header="infer").values
x = data[:,1:-1]
y = data[:,-1]
test_split = float(input("Enter the test_size : "))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split,stratify=y)
k =int(input("Enter the number of clusters you want to divide your data in , i.e , k : "))
n = int(input("Enter the number of iterationd you want to run the algorithm : "))
centroids = np.zeros(shape=(k,x_train.shape[1]))
per = np.random.permutation(x_train.shape[0])
for i in range(k) : 
    centroids[i,:] = x_train[per[i],:]
for it in range(n) : 
    dist = np.zeros(shape =(k,x_train.shape[0]))
for i in range(k) :
    dist[i,:] = np.sqrt(np.sum((x_train-centroids[i,:])**2,axis=1))
    membership = np.argmin(dist,axis=0)
for i in range(k) : 
    centroids[i,:] = np.mean(x_train[membership==i,:],axis=0)
print("Centroids after" + str(n) + "iterations : ")
print(centroids)
dist = np.zeros(shape=(k,x_train.shape[0]))
for i in range(k) : 
    dist[i] = np.sqrt(np.sum((x_train-centroids[i])**2,axis=1))
print(y_test.astype(int))
print(membership)
