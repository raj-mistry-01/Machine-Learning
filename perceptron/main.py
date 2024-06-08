import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
ds1=np.random.randint(low=1,high=10,size=(50,2))
ds2=-ds1
X=np.concatenate((ds1,ds2),axis=0)
Y=np.ones(shape=100)
Y[:50]=0
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y)
plt.scatter(x=X_train[:,0],y=X_train[:,1],c=Y_train)
plt.show()
n_samples=X_train.shape[0]
n_features=X_train.shape[1]
w=np.random.uniform(0,1,size=n_features)
b=np.random.uniform(0,1,1)
n_epoch=int(input("ENTER THE NUMBER OF EPOCHS:"))
lr=0.01
for e in range(n_epoch):
    for s in range(n_samples):
        net=np.dot(X_train[s,:],w)+b
        if net>=0:
            a=1
        else:
            a=0
        error=Y_train[s]-a
        w=w+lr*error*X_train[s,:]
        b=b+lr*error
net=np.dot(X_test,w)+b
pred=list(map(int,(net>=0)))
print(pred)
print("CLASSIFICATION REPORT:")
print(classification_report(y_true=Y_test,y_pred=pred))
m=-w[0]/w[1]
c=-b/w[1]
def plot_decision_boundary(X):
    for x in np.linspace(np.min(X[:,0]),np.max(X[:,0])):
        y=m*x+c
        plt.plot(x,y,linestyle='-',color='k',marker='.')
    plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
    plt.show()
plot_decision_boundary(X_train)  
