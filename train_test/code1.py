# with library
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"csvs\BostonHousing.csv").values
x = df[:,:-1]
y = df[:,-1]
x = df[:,:-1]
y = df[:,-1]
nrows = df.shape[0]
print("Total rows : ",nrows)
split_size = float(input("Enter a number between 0 and 1 : "))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=split_size)
print("Shapes : ",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
