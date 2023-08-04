from deep import model,predict
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split as tts
data=sio.loadmat('data.mat')


seed=1
x_train=data['X'].T
y_train=data['y'].T
x_test=data['Xval'].T
y_test=data['yval'].T

# x_train,x_test,y_train,y_test=tts(x_data,y_data,train_size=0.7,random_state=1)

# x_train,x_test,y_train,y_test=x_train.T,x_test.T,y_train.T,y_test.T

print(x_train.shape,x_test.shape,y_train.shape)

parameters=model(x_train,y_train,layer_dims=[x_train.shape[0],20,3,1],learning_rate=0.5,seed=seed,num_of_iteration=30000,lamda_=0.6,keep_prob=1)

# print(parameters['w1'][4,])

y_hat=predict(x_test,parameters)
print(y_hat)
print("test Accuracy:",np.mean(y_hat==y_test)*100,"%")

y_train_hat=predict(x_train,parameters)
print("Train accuracy:",np.mean(y_train_hat==y_train)*100,"%")
