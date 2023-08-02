import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import exp

np.random.seed(1)

def initialize_parameters(n_x,n_h,n_y):
    '''
    n_x = size of input layer
    n_h = size of the hidden layer
    n_y = size of the output layer
    
    Return a dictionary with initialized weights
    
    '''

    W1= np.random.randn(n_h,n_x)*0.01
    W2= np.random.randn(n_y,n_h)*0.01
    b1= np.zeros((n_h,1))
    b2= np.zeros((n_y,1))

    assert(W1.shape==(n_h,n_x))
    assert(W2.shape==(n_y,n_h))
    assert(b1.shape==(n_h,1))
    assert(b2.shape==(n_y,1))


    parameters={
        'W1':W1,
        'W2':W2,
        'b1':b1,
        'b2':b2
    }

    return parameters


# layer_dims = number of units in each layer, a list eg [3,4,4,1]

def initialize_parameters_deep(layer_dims):

    parameters=dict()

    L=len(layer_dims)

    for l in range(1,L):
        parameters['W'+str(l)]=np.random.rand(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

        assert(parameters['W'+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
        assert(parameters['b'+str(l)].shape==(layer_dims[l],1))

    return parameters

#Forward Propagation Module


def linear_forward(A,W,b):

    #dimension of Z will be no of units X no. of inputs

    Z=np.dot(W,A)+b
    cache=(A,W,b)
    return Z,cache

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))

    return A,Z

def relu(Z):
    A=np.where(Z>0,Z,0)
    return A,Z

def linear_activation_forward(A_prev,W,b,activation='relu'):

    Z,linear_cache=linear_forward(A_prev,W,b)

    if activation=='relu':
        A,activation_cache=relu(Z)
    else:
        A,activation_cache=sigmoid(Z)
    
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activation_cache)

    return A,cache

def L_model_forward(X,parameters):
    # caches will be list of tuple
    caches=[]
    A=X
    L=len(parameters)//2 #Floored division

    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)])
        caches.append(cache)

    Al,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)

    return Al,caches

def compute_cost(Al,Y):

    m=Y.shape[1]

    cost=-np.sum(Y*np.log(Al)+(1-Y)*np.log(1-Al))/m
    cost=np.squeeze(cost)

    assert(cost.shape==())
    return cost

def linear_backward(dZ,cache):

    A_prev,W,b=cache
    m=A_prev.shape[1]

    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)  

    return dA_prev,dW,db


def sigmoid_backward(dA,activation_cache):
    
    s=1/(1+np.exp(-activation_cache))
    dz=dA*s*(1-s)

    return dz

def relu_backward(dA, activation_cache):

    dz=np.where(dA>0,dA,0)
    return dz


def linear_activation_backward(dA,cache,activation='relu'):

    linear_cache,activation_cache=cache

    if activation=='sigmoid':
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,dB=linear_backward(dZ,linear_cache)
        return dA_prev,dW,dB
    
    dZ=relu_backward(dA,activation_cache)
    dA_prev,dW,dB=linear_backward(dZ,linear_cache)
    return dA_prev,dW,dB

def L_model_backward(AL,Y,caches):

    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)

    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    
    grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=linear_activation_backward(dAL,caches[L-1],'sigmoid')

    for l in reversed(range(L-1)):
        dA_prev_temp=grads['dA'+str(l+1)]
        cache=caches[l]
        dA_prev,dW,dB=linear_activation_backward(dA_prev_temp,cache)
        grads['dA'+str(l)]=dA_prev
        grads['dW'+str(l+1)]=dW
        grads['db'+str(l+1)]=dB
    
    return grads

def update_parameters(parameters,grads,learning_rate):

    L=len(parameters)//2 #layers in neural network

    for l in range(L):
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)]=parameters['b'+str(l+1)]-learning_rate*grads['db'+str(l+1)]

X=np.array([[2,3,0,1,-1],[3,-3,4,1,4]])

A,activation_cache=relu(X)
print(A,'\n',activation_cache)

print(SMALL)


# parameters=initialize_parameters_deep([5,4,3])

# print(parameters['W1'],parameters['W2'],parameters['b2'])