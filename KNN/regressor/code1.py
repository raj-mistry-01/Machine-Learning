#with library
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv(r"csvs\Iris.csv", header='infer').values

x = data[:,:-1] 
y = data[:, -1] 

test_split = float(input("Enter a number between 0 to 1 to specify how much data is required as the test data: "))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
k = int(input("Enter the number of nearest neighbours to be used, i.e. k: "))
dist = np.zeros(shape=x_train.shape[0])
pred = np.zeros(shape=x_test.shape[0])
model=KNeighborsRegressor(n_neighbors=k, weights='distance')
model.fit(x_train,y_train)
pred=model.predict(x_test)
mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
print("Using Sklearn:")
print("MAE:",mae)
print("MSE:",mse)
