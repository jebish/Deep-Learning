#Mini Batch Adam Optimizer

#Imports
import numpy as np
from matplotlib import pyplot as plt

#constants
E=10**(-8)
LE=0.01

def mini_batch(x,y,batch_size):
    m=y.shape[1]
    mini_batches=list() #Will make mini batches as list of tuples

    #shuffling
    permutation=np.random.permutation(m)
    x_reshuffled=x[:,permutation]
    y_reshuffled=y[:,permutation]
    
    assert(x_reshuffled.shape==x.shape)
    assert(y_reshuffled.shape==y.shape)

    #creating mini batches
    num_of_batches=m//batch_size
    
    for i in range(num_of_batches):
        x_batch=x_reshuffled[:,i*batch_size:(i+1)*batch_size]
        y_batch=y_reshuffled[:,i*batch_size:(i+1)*batch_size]
        mini_batch=(x_batch,y_batch)
        mini_batches.append(mini_batch)
    
    if not m%batch_size==0:
        x_batch=x_reshuffled[:,num_of_batches*batch_size:] 
        y_batch=y_reshuffled[:,num_of_batches*batch_size:]
        mini_batch=(x_batch,y_batch)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_weights(layer_dims):
    num_of_layers=len(layer_dims)
    parameters,velocity,displacement=dict(),dict(),dict()
    for l in range(1,num_of_layers):
        parameters['w'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

        assert(parameters['w'+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
        assert(parameters['b'+str(l)].shape==(layer_dims[l],1))

        velocity['vdw'+str(l)]=np.zeros(parameters['w'+str(l)].shape)
        velocity['vdb'+str(l)]=np.zeros(parameters['b'+str(l)].shape)
        displacement['sdw'+str(l)]=np.zeros(parameters['w'+str(l)].shape)
        displacement['sdb'+str(l)]=np.zeros(parameters['b'+str(l)].shape)
    
    
    return parameters,velocity,displacement


def forward_linear(a_prev,w,b):
    z=np.dot(w,a_prev)+b
    linear_cache=(a_prev,w,b)
    return z,linear_cache

def lrelu(z):
    s=np.where(z>0,z,LE*z)
    return s,z

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s,z

def forward_activation(a_prev,w,b,activation='lrelu'):
    z,linear_cache=forward_linear(a_prev,w,b)
    if activation=='sigmoid':
        a,activation_cache=sigmoid(z)
    elif activation=='lrelu':
        a,activation_cache=lrelu(z)
    cache=(activation_cache,linear_cache)
    return a,cache

def forward_model(x,parameters):
    num_of_layers=len(parameters)//2

    a_prev=x
    caches=list() #list of tuples
    for l in range(1,num_of_layers):
        a_prev,cache=forward_activation(a_prev,parameters['w'+str(l)],parameters['b'+str(l)],activation='lrelu')
        caches.append(cache)
    #output layer
    al,cache=forward_activation(a_prev,parameters['w'+str(num_of_layers)],parameters['b'+str(num_of_layers)],activation='sigmoid')
    caches.append(cache)

    return al,caches

def compute_cost(al,y):
    m=y.shape[1]
    cost=(-1./m)*(np.dot(y,np.log(al.T+E))+np.dot(1-y,np.log(1-al.T+E)))
    cost=np.squeeze(cost)
    return cost

def backward_linear(dz,linear_cache):

    m=dz.shape[1]
    a_prev,w,b=linear_cache
    dw=np.dot(dz,a_prev.T)/m
    db=np.mean(dz,axis=1,keepdims=True)
    da_prev=np.dot(w.T,dz)

    return da_prev,dw,db

def sigmoid_derivative(da,z):
    s=sigmoid(z)
    dz=da*s*(1-s)
    return dz

def lrelu_derivative(da,z):
    dz=np.where(z>0,da,LE*da)
    return dz

def backward_model(al,y,caches):

    grads={}
    L=len(caches)
    dz=al-y
    cache=caches[L-1]
    (activation_cache,linear_cache)=cache
    da_prev,dw,db=backward_linear(dz,linear_cache)
    grads['dw'+str(L)]=dw
    grads['db'+str(L)]=db

    for l in reversed(range(L-1)):
        cache=caches[l]
        (activation_cache,linear_cache)=cache
        dz=lrelu_derivative(da_prev,activation_cache)
        da_prev,grads['dw'+str(l+1)],grads['db'+str(l+1)]=backward_linear(dz,linear_cache)

    return grads

def update_parameters(parameters,grads,learning_rate):
    num_of_layers=len(parameters)//2
    for l in range(1,num_of_layers+1):
        parameters['w'+str(l)]=parameters['w'+str(l)]-learning_rate*(grads['dw'+str(l)])
        parameters['b'+str(l)]=parameters['b'+str(l)]-learning_rate*(grads['db'+str(l)])
    
    return parameters

def update_parameters_adam(parameters,grads,velocity,displacement,learning_rate,t,beta_1,beta_2):

    num_of_layers=len(parameters)//2

    velocity_corrected={}
    displacement_corrected={}

    for l in range(1,num_of_layers+1):
        velocity['vdw'+str(l)]=beta_1*velocity['vdw'+str(l)]+(1-beta_1)*grads['dw'+str(l)]
        velocity_corrected['vdw'+str(l)]=velocity['vdw'+str(l)]/(1-(beta_1**t))
        velocity['vdb'+str(l)]=beta_1*velocity['vdb'+str(l)]+(1-beta_1)*grads['db'+str(l)]
        velocity_corrected['vdb'+str(l)]=velocity['vdb'+str(l)]/(1-(beta_1**t))
        displacement['sdw'+str(l)]=beta_2*displacement['sdw'+str(l)]+(1-beta_2)*(np.square(grads['dw'+str(l)]))
        displacement_corrected['sdw'+str(l)]=displacement['sdw'+str(l)]/(1-(beta_2**t))
        displacement['sdb'+str(l)]=beta_2*displacement['sdb'+str(l)]+(1-beta_2)*(np.square(grads['db'+str(l)]))
        displacement_corrected['sdb'+str(l)]=displacement['sdb'+str(l)]/(1-(beta_2**t))

        parameters['w'+str(l)]=parameters['w'+str(l)]-((learning_rate*velocity['vdw'+str(l)])/np.sqrt(displacement_corrected['sdw'+str(l)]+E))
        parameters['b'+str(l)]=parameters['b'+str(l)]-((learning_rate*velocity['vdb'+str(l)])/np.sqrt(displacement_corrected['sdb'+str(l)]+E))
        # parameters['w'+str(l)]=parameters['w'+str(l)]-((learning_rate*velocity_corrected['vdw'+str(l)]))
        # parameters['b'+str(l)]=parameters['b'+str(l)]-((learning_rate*velocity_corrected['vdb'+str(l)]))
    
    if t==1:
        # print(velocity['vdw1']/np.sqrt(displacement['sdw1']))
        print(velocity['vdw2']==grads['dw2'])
    return parameters,velocity,displacement



def model(x,y,layer_dims,num_of_iteration=10,learning_rate=0.07,batch_size=64,beta_1=0.9,beta_2=0.998):

    parameters,velocity,displacement=initialize_weights(layer_dims=layer_dims)
    costs=list()
    x_train,y_train=x,y
    t=0
    decay=0.01
    learning_rate_initial=learning_rate
    for i in range(num_of_iteration):
        learning_rate=learning_rate_initial/(1+i*decay)
        mini_batches=mini_batch(x=x_train,y=y_train,batch_size=batch_size)
        num_of_batch=len(mini_batches)
        for l in range(num_of_batch):
            (x,y)=mini_batches[l]
            al,caches=forward_model(x,parameters=parameters)

            cost=compute_cost(al,y)
        
            costs.append(cost)
            if (i%100==0):
                print(cost)

            grads=backward_model(al,y,caches)
            t+=1
            # parameters=update_parameters(parameters=parameters,grads=grads,learning_rate=learning_rate)
            parameters,velocity,displacement=update_parameters_adam(parameters,grads,velocity=velocity,displacement=displacement,learning_rate=learning_rate,t=t,beta_1=beta_1,beta_2=beta_2)

    plt.plot(costs[::10])
    plt.show()

    return parameters


def predict(x,parameters):

    y_hat,caches=forward_model(x,parameters=parameters)
    y_hat=(y_hat>=0.5).astype(int)
    return y_hat

    
    
        
