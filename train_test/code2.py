# without library
import numpy as np
import pandas as pd
import math
df = pd.read_csv(r"csvs\BostonHousing.csv").values
x = df[:,:-1]
y = df[:,-1]
nrows = df.shape[0]
print("Total rows :",nrows)
test_splits = float(input("Enter a number between 0 and 1 : "))
nrows_train = math.floor((1-test_splits)*nrows)
all_indices = np.random.permutation(nrows)
x_train = x[all_indices[0:nrows_train],:]
y_train = y[all_indices[0:nrows_train]]
x_test = x[all_indices[nrows_train:],:]
y_test = y[all_indices[nrows_train:]]
print("Shapes : ",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print("Unions : ",len(set(all_indices[0:nrows]).union(all_indices[nrows_train:])))
print("Intersection : ",len(set(all_indices[0:nrows_train]).intersection(all_indices[nrows_train:])))
