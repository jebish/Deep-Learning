#building deep neural network from scratch using gradient descent

#import libraries

import numpy as np
from matplotlib import pyplot as plt

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

def linear_activation(a_prev,w,b,activation):
    z,linear_cache=linear_forward(a_prev,w,b)
    if activation=='sigmoid':
        a,activation_cache=sigmoid(z)
    elif activation=='relu':
        a,activation_cache=relu(z)
    
    assert(a.shape==(w.shape[0],a_prev.shape[1]))
    cache=(linear_cache,activation_cache)
    return a,cache

# #forward_model
def forward_model(X,parameters):

    caches=list()
    L=len(parameters)//2
    a=X
    for l in range(1,L):
        a_prev=a
        a,cache=linear_activation(a_prev,parameters['w'+str(l)],parameters['b'+str(l)],activation='relu')
        caches.append(cache)
        
    
    aL,cache=linear_activation(a,parameters['w'+str(L)],parameters['b'+str(L)],activation='sigmoid')
    caches.append(cache)

    return aL,caches

# #cost calculation
def cost_compute(y,a):
    m=y.shape[1]
    cost=(1./m) * (-np.dot(y,np.log(a).T) - np.dot(1-y, np.log(1-a).T))
    cost=np.squeeze(cost)
    return cost

# def linear_forward(A, W, b):
#     """
#     Implement the linear part of a layer's forward propagation.

#     Arguments:
#     A -- activations from previous layer (or input data): (size of previous layer, number of examples)
#     W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
#     b -- bias vector, numpy array of shape (size of the current layer, 1)

#     Returns:
#     Z -- the input of the activation function, also called pre-activation parameter 
#     cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
#     """
    
#     Z = W.dot(A) + b
    
#     assert(Z.shape == (W.shape[0], A.shape[1]))
#     cache = (A, W, b)
    
#     return Z, cache

# def linear_activation_forward(A_prev, W, b, activation):
#     """
#     Implement the forward propagation for the LINEAR->ACTIVATION layer

#     Arguments:
#     A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
#     W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
#     b -- bias vector, numpy array of shape (size of the current layer, 1)
#     activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

#     Returns:
#     A -- the output of the activation function, also called the post-activation value 
#     cache -- a python dictionary containing "linear_cache" and "activation_cache";
#              stored for computing the backward pass efficiently
#     """
    
#     if activation == "sigmoid":
#         # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
#         Z, linear_cache = linear_forward(A_prev, W, b)
#         A, activation_cache = sigmoid(Z)
    
#     elif activation == "relu":
#         # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
#         Z, linear_cache = linear_forward(A_prev, W, b)
#         A, activation_cache = relu(Z)
    
#     assert (A.shape == (W.shape[0], A_prev.shape[1]))
#     cache = (linear_cache, activation_cache)

#     return A, cache

# def L_model_forward(X, parameters):
#     """
#     Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
#     Arguments:
#     X -- data, numpy array of shape (input size, number of examples)
#     parameters -- output of initialize_parameters_deep()
    
#     Returns:
#     AL -- last post-activation value
#     caches -- list of caches containing:
#                 every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
#                 the cache of linear_sigmoid_forward() (there is one, indexed L-1)
#     """

#     caches = []
#     A = X
#     L = len(parameters) // 2                  # number of layers in the neural network
    
#     # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
#     for l in range(1, L):
#         A_prev = A 
#         A, cache = linear_activation_forward(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], activation = "relu")
#         caches.append(cache)
    
#     # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
#     AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
#     caches.append(cache)
    
#     assert(AL.shape == (1,X.shape[1]))
            
#     return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

#backward_linear
# def backward_linear(dz,linear_cache):
    
#     a_prev,w,b=linear_cache
#     m=a_prev.shape[1]
#     dw=np.dot(dz,a_prev.T)/m
#     db=(np.sum(dz,axis=1,keepdims=True))/m
#     da_prev=np.dot(w.T,dz)
#     return da_prev,dw,db



#backward_linear_activation
# def sigmoid_derivative(da,activation_cache):
#     s=1/(1+np.exp(-activation_cache))
#     dz=da*s*(1-s)
#     return dz

# def relu_derivative(da,activation_cache):
#     return np.where(da>0,da,0)

# def backward_activation(da_prev,cache,activation='relu'):
#     linear_cache,activation_cache=cache
#     if activation=='sigmoid':
#         dz=sigmoid_derivative(da_prev,activation_cache)
#     elif activation=='relu':
#         dz=relu_derivative(da_prev,activation_cache)
    
#     da_prev,dw,db=backward_linear(dz,linear_cache)
#     return da_prev,dw,db

#backward_model
# def backward_model(y,al,caches):
#     # L=len(caches)
#     # grads={}
#     # cache=caches[L-1]
#     # da_prev=-(np.divide(y,al)-np.divide(1-y,1-al))
#     # da_prev,grads['dw'+str(L)],grads['db'+str(L)]=backward_activation(da_prev,cache,activation='sigmoid')
    
#     # for l in reversed(range(L-1)):
#     #     cache=caches[l]
        
#     #     da_prev_temp,grads['dw'+str(l+1)],grads['db'+str(l+1)]=backward_activation(da_prev,cache,activation='relu')
#     #     da_prev=da_prev_temp
        
#     # return grads 

#     grads = {}
#     L = len(caches) # the number of layers
#     m = al.shape[1]
#     y = y.reshape(al.shape) # after this line, Y is the same shape as AL
    
#     # Initializing the backpropagation
#     dal = - (np.divide(y, al) - np.divide(1 - y, 1 - al))
    
#     # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
#     current_cache = caches[L-1]
#     grads["da" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = backward_activation(dal, current_cache, activation = "sigmoid")
    
#     for l in reversed(range(L-1)):
#         # lth layer: (RELU -> LINEAR) gradients.
#         current_cache = caches[l]
#         da_prev_temp, dw_temp, db_temp = backward_activation(grads["da" + str(l + 2)], current_cache, activation = "relu")
#         grads["da" + str(l + 1)] = da_prev_temp
#         grads["dw" + str(l + 1)] = dw_temp
#         grads["db" + str(l + 1)] = db_temp

#     return grads

#backward from net

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dw" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#update_parameters
# def update_parameters(parameters,grads,learning_rate):
#     L=len(parameters)//2
#     for l in range(L):
#         parameters['w'+str(l+1)]=parameters['w'+str(l+1)]-(learning_rate*grads['dw'+str(l)])
#         parameters['b'+str(l)]=parameters['b'+str(l)]-(learning_rate*grads['db'+str(l)])

#     return parameters

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

#model
def model(x,y,layer_dims,learning_rate=0.3,seed=1,num_of_iteration=100):

    parameters=initialize_parameters(layer_dims,seed)

    costs=[] #for cost history
    grads={}

    for i in range(num_of_iteration):

        #forward prop
        # al,caches=forward_model(x,parameters)

        al,caches=forward_model(x,parameters)

        #cost calculation
        # cost=cost_compute(y,al)
        cost=compute_cost(al,y)

        costs.append(cost)
        #backprop

        # grads=backward_model(y,al,caches)

        grads=L_model_backward(AL=al,Y=y,caches=caches)

        #update parameters
        parameters=update_parameters(parameters,grads,learning_rate)

        if i%100==0:
            print (costs[i])
    
    plt.plot(np.squeeze(costs))
    plt.show()
    return parameters

def predict(X,parameters):
    y_hat=forward_model(X,parameters)

#call below with loading data