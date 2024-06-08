# with library
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.neighbors import KNeighborsClassifier
data=pd.read_csv(r"csvs\Iris.csv",header='infer').values 
X=data[:,1:-1] 
y=data[:,-1] 
split_size = float(input("Enter a number between 0 and 1 as the train data spilting : "))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=split_size,stratify=y)
k=int(input("Enter the number of nearest neighbours to be used, i.e. k:"))
model=KNeighborsClassifier(n_neighbors=k, weights='distance')
model.fit(X_train,y_train)
pred=model.predict(X_test)
accuracy=accuracy_score(y_test,pred)
print("Accuracy:", accuracy)
print(classification_report(y_test,pred))
