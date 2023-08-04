from deep import model,predict
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


print(y_train.shape)

parameters=model(x_train,y_train,layer_dims=[x_train.shape[0],20,7,5,1],num_of_iteration=3000,learning_rate=0.0075)

