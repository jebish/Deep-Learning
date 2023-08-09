from adam import model,predict
import numpy as np
import h5py
from matplotlib import pyplot as plt

np.random.seed(1)

data=h5py.File("train_catvnoncat.h5",'r')
x_train=np.array(data['train_set_x'],dtype='float64')
y_train=np.array(data['train_set_y'])
m=y_train.shape[0]
x_train=x_train.reshape(m,-1).T
x_train=x_train/255

y_train=y_train.reshape(m,-1).T

data1=h5py.File('test_catvnoncat.h5','r')

x_test=np.array(data1['test_set_x'])
y_test=np.array(data1['test_set_y'])

x_test=x_test.reshape(x_test.shape[0],-1).T
x_test=x_test/255
y_test=y_test.reshape(y_test.shape[0],-1).T


# print(y_train.shape)

parameters=model(x_train,y_train,layer_dims=[x_train.shape[0],20,7,5,1],num_of_iteration=1000,learning_rate=0.0075,batch_size=64,beta_1=0.9)

y_test_hat=predict(x_test,parameters=parameters)
y_train_hat=predict(x_train,parameters=parameters)
print("Accuracy on training set is: ",np.mean(y_train_hat==y_train)*100,"%")
print("Accuracy on test set is: ",np.mean(y_test_hat==y_test)*100,"%")


# parameters=model(x_train,y_train,layer_dims=[x_train.shape[0],20,7,5,1],num_of_iteration=3000,learning_rate=0.0075,keep_prob=1)

