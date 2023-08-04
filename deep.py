#building deep neural network from scratch using gradient descent

#import libraries

import numpy as np
from matplotlib import pyplot as plt

E = 10**(-9)
# initialize parameters
def initialize_parameters(layer_dims,seed):
    
    np.random.seed(seed)
    parameters={}
    L=len(layer_dims)
    for l in range(1,L):
        parameters['w'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])/np.sqrt(layer_dims[l-1]) 
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == layer_dims[l], 1)
    
    return parameters

def initialize_parameters_shallow(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     


#linear_forward
def linear_forward(a_prev,w,b):
    z=np.dot(w,a_prev)+b
    cache=(a_prev,w,b)
    return z,cache


# #linear_forward_activation
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s,z


def relu(z):
    s=np.maximum(0,z)
    return s,z

def circular(z):
    r_2=np.mean(z**2)
    s=(r_2-z**2)
    return s,z


def linear_activation(a_prev,w,b,activation,keep_prob):
    z,linear_cache=linear_forward(a_prev,w,b)
    if activation=='sigmoid':
        a,activation_cache=sigmoid(z)
    elif activation=='relu':
        a,activation_cache=relu(z)
    elif activation=='circular':
        a,activation_cache=circular(z)
    d=np.random.rand(a.shape[0],a.shape[1])
    d=(d<=keep_prob).astype(int)
    a=a*d
    a=a/np.mean(d)
    assert(a.shape==(w.shape[0],a_prev.shape[1]))
    cache=(linear_cache,d,activation_cache)
    
    return a,cache

# #forward_model
def forward_model(X,parameters,keep_prob):

    caches=list()
    L=len(parameters)//2
    a=X
    for l in range(1,L):
        a_prev=a
        a,cache=linear_activation(a_prev,parameters['w'+str(l)],parameters['b'+str(l)],activation='relu',keep_prob=keep_prob)
        caches.append(cache)
        
    
    aL,cache=linear_activation(a,parameters['w'+str(L)],parameters['b'+str(L)],activation='sigmoid',keep_prob=1)
    caches.append(cache)

    aL.dtype='float64'
    return aL,caches

# #cost calculation
def cost_compute(y,a):
    m=y.shape[1]
    np.seterr(invalid='ignore')
    cost=(1./m) * (-np.dot(y,np.log(a+E).T) - np.dot(1-y, np.log(1-a+E).T))
    cost=np.squeeze(cost)
    return cost

def cost_compute_L2(y,a,parameters,lamda_):
    m=y.shape[1]
    L=len(parameters)//2
    regularization_cost=0
    for l in range(1,L+1):
        regularization_cost=regularization_cost+np.sum(parameters['w'+str(l)]**2)
    regularization_cost=(regularization_cost*lamda_)/(2*m)
    cost=(1./m) * (-np.dot(y,np.log(a+E).T) - np.dot(1-y, np.log(1-a+E).T))+regularization_cost    
    cost=np.squeeze(cost)
    return cost

# backward_linear
def backward_linear(dz,linear_cache):
    
    a_prev,w,b=linear_cache
    m=a_prev.shape[1]
    dw=np.dot(dz,a_prev.T)/m
    db=(np.sum(dz,axis=1,keepdims=True))/m
    da_prev=np.dot(w.T,dz)
    return da_prev,dw,db



#backward_linear_activation
def sigmoid_derivative(da,activation_cache):
    s=1/(1+np.exp(-activation_cache))
    dz=da*s*(1-s)
    return dz

def relu_derivative(da,activation_cache):
    return np.where(activation_cache>0,da,0)

def circular_derivative(da,activation_cache):
    s=-(2*activation_cache)
    dz=da*s
    return dz

def backward_activation(da_prev,cache,keep_prob,activation='relu'):
    linear_cache,d,activation_cache=cache
    da_prev=da_prev*d
    da_prev=da_prev/np.mean(d)
    if activation=='sigmoid':
        dz=sigmoid_derivative(da_prev,activation_cache)
    elif activation=='relu':
        dz=relu_derivative(da_prev,activation_cache)
    elif activation=='circular':
        dz=relu_derivative(da_prev,activation_cache)
    
    
    da_prev,dw,db=backward_linear(dz,linear_cache)
    
    return da_prev,dw,db

# backward_model
def backward_model(y,al,caches,keep_prob):
    L=len(caches)
    grads={}
    cache=caches[L-1]
    # da_prev=-(np.divide(y,al+E)-np.divide(1-y,1-al+E))
    da_prev=np.zeros((y.shape))
    da_prev=-(np.divide(y,al,where=al!=0)-np.divide(1-y,1-al,where=(1-al)!=0))
    da_prev,grads['dw'+str(L)],grads['db'+str(L)]=backward_activation(da_prev,cache,activation='sigmoid',keep_prob=1)
    
    for l in reversed(range(L-1)):
        cache=caches[l]
        
        da_prev_temp,grads['dw'+str(l+1)],grads['db'+str(l+1)]=backward_activation(da_prev,cache,activation='relu',keep_prob=keep_prob)
        da_prev=da_prev_temp
        
    return grads 

   

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["w" + str(l+1)] = parameters["w" + str(l+1)] - learning_rate * grads["dw" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def update_parameters_L2(parameters, grads, learning_rate,lamda_,m):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        weight_decay=parameters['w'+str(l+1)]*lamda_/m
        parameters["w" + str(l+1)] = parameters["w" + str(l+1)] - learning_rate * grads["dw" + str(l+1)] - learning_rate *weight_decay
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


#model
def model(x,y,layer_dims,keep_prob,learning_rate=0.3,seed=1,num_of_iteration=100,lamda_=0.3):

    parameters=initialize_parameters(layer_dims,seed)
    print(parameters)
    costs=[] #for cost history
    grads={}
    m=y.shape[1]
    L2=0
    for i in range(num_of_iteration):

        #forward prop
        # al,caches=forward_model(x,parameters)

        al,caches=forward_model(x,parameters=parameters,keep_prob=keep_prob)

        #cost calculation
        if L2==1:
            cost=cost_compute_L2(y,al,parameters,lamda_)
        else:
            cost=cost_compute(y,al)
        

        costs.append(cost)
        #backprop

        grads=backward_model(y,al,caches,keep_prob=keep_prob)

        # grads=L_model_backward(AL=al,Y=y,caches=caches)

        #update parameters
        
        if L2==1:
            parameters=update_parameters_L2(parameters,grads,learning_rate,lamda_=lamda_,m=m)
        else:
            parameters=update_parameters(parameters,grads,learning_rate)
            

        if i%1000==0:
            print (costs[i])
    
    plt.plot(np.squeeze(costs))
    plt.show()
    return parameters

def predict(X,parameters):
    y_hat,caches=forward_model(X,parameters,keep_prob=1)
    y_hat=(y_hat>0.5).astype(int)
    return y_hat

#call below with loading data