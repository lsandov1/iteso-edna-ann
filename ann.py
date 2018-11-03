#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Classiffying Dogs vs Cats datatset
# 
# There are 2000 images for training and 1000 images for testing.
# 
# If you want to explore original dataset images: 
# https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
# 
# Dataset as hdf5 format:
# https://www.dropbox.com/sh/ca1o3f4p0kui4ds/AABB0IppedeA1BiZu7jMb2-ka?dl=0

# In[3]:


data = pd.read_hdf('train.h5', 'df') #Load the data
data = data.sample(frac = 1)


# ## Show random images

# In[4]:


idx = np.random.choice(100, 5)
img = 1
plt.subplots(1, 5, figsize=(16, 10))
for i in idx:
    ax = plt.subplot(1, 5, img)
    ax.axis('off')
    ax.imshow(data['image'].iloc[i])
    img += 1


# ## Preprocess the data

# In[5]:


images = data['image'].values
y = data['label'].values
m = images.shape[0]
print('images', m)
print('y', y.shape)


# ### Get the arrays and scale the data

# In[6]:


X = np.zeros((m, 100*100*3))
for i in range(m):
    X[i] = images[i].flatten().reshape(1, -1)

X = X / 255
y = y.reshape(-1, 1)
print('X', X.shape)


# ## The model
# 
# Cost function for the neural network:
# 
# $$
# J(w) = \frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} \biggl( -y_{ik}\  log(\sigma_k(x_i)) - (1 - y_{ik})\ log(1 - \sigma_k(x_i)) \biggr)
# $$
# 
# $K$ is the number of output units, $\sigma_k(x_i)$ is the activation of sample $i$ on neuron $k$.
# 
# ### Backpropagation algorithm
# 
# **Feed-forward in a vectorized form**
# 
# 1. Compute the weithed sum $z_{l+1} = a_{l} w_{l}^T + b_{l}^T$
# 1. Apply the activation function $a_{l+1} = \sigma(z_{l+1})$
# 
# **Backward-propagation**
# 
# 1. For output layer $\delta_l = a_l - y$
# 2. For hidden layer $\delta_l = \sigma'(z_l) * \delta_{l+1} w_l$
# 
# **Gradients**
# 
# 1. For bias term, $\frac{\partial}{\partial b_l} = \frac{1}{m}sum(\delta_{l+1})$, recall that shape of $\frac{\partial}{\partial b_l}$ must match the shape of $b_l$
# 2. For weights, $\frac{\partial}{\partial w_l} = \frac{1}{m} \delta_{l+1} a_l$
# 
# ### Xavier initialization
# 
# Their major goal is to prevent gradient vanishing and too-large weight problems, the formula for a normal distribution is as follows:
# 
# $$
# W_l = randn(n_l, n_{l-1}) * \sqrt{2 / (n_{l-1} + n_l)}
# $$

# In[7]:


# nx, number of features
# nh, number of hidden neurons
# ny, number of output neurons
def Xavier_init_w(nx, nh, ny):
    np.random.seed(2)
    # w's will be created randomly
    # b's will be zeros
    W1 = np.random.randn(nh, nx) * ((2/(nh + nx))**(0.5))
    b1 = np.zeros(shape=(1, nh))
    W2 = np.random.randn(ny, nh) * ((2/(ny + nh))**(0.5))
    b2 = np.zeros(shape=(ny, 1))    
    
    W = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return W

# Activation function
def sigma(z, activation):
    if activation == 'sigmoid':
        out = 1 / (1 + np.exp(-z))
    else:
        if activation == 'tanh':
            out = (np.exp(z)-np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return out

# Implement forward propagation to calculate output probabilities
def forward(x, W):
    W1 = W['W1']
    b1 = W['b1']
    W2 = W['W2']
    b2 = W['b2']

    Z2 = np.dot(x, W1.T) + b1
    A2 = sigma(Z2, 'tanh')
    Z3 = np.dot(A2, W2.T) + b2
    A3 = sigma(Z3, 'sigmoid')
    
    Z = {"Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, Z

def cost(A, y):
    m = len(y)
    E = (1/m) * np.sum(-y * np.log(A) - ((1 - y) * np.log(1 - A)))
    return E
    

def d_sigma(a, activation):
    if activation == 'tanh':
        out = 1 - (sigma(a, activation)**2)
    else:
        if activation == 'sigmoid':
            out = sigma(a, activation) * (1 - sigma(a, activation))
    return out

def backward(W, Z, x, y):
    m = x.shape[1]
    
    W1 = W['W1']
    W2 = W['W2']
    A2 = Z['A2']
    A3 = Z['A3']
    Z2 = Z['Z2']
    Z3 = Z['Z3']
    
    d3 = (A3 - y)
    dW2 = (1/m) * np.dot(d3.T, A2)
    db2 = (1/m) * np.sum(d3)
    d2 = d_sigma(Z2, 'tanh') * np.dot(d3,W2)
    dW1 = (1 /m) * np.dot(d2.T,X)
    db1 = (1 / m) * np.sum(d2,axis = 0)    
    grad = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2} 
        
    return grad

def predict(x, W):
    A3, Z = forward(x, W)
    y_hat = list(map(lambda x: 1 if x > 0.5 else 0, A3))
    y_hat = np.array(y_hat)
    y_hat = y_hat.reshape(-1,1)
    return y_hat

def accuracy(y_hat, y):
    m = len(y)
    tptn = (y == y_hat).sum()
    acc = tptn / m
    return acc


# In[8]:


def model(x, y, nh, alpha = 0.1, epochs = 10000, debug = False):
    np.random.seed(2)
    nx = x.shape[1]
    ny = 1
    
    W = Xavier_init_w(nx, nh, ny)
    
    A3, Z = forward(x, W)
    print('Initial cost:', cost(A3, y))
    
    J= {}
    for i in range(epochs):         
        A3, Z = forward(x, W)       
        J[i] = cost(A3, y)
        grad = backward(W, Z, X, y)
     
        #update w's and b's here
        W['W1'] = W['W1'] - alpha * grad['dW1']
        W['W2'] = W['W2'] - alpha * grad['dW2']
        W['b1'] = W['b1'] - alpha * grad['db1']
        W['b2'] = W['b2'] - alpha * grad['db2']
        
        if i % 200 == 0 and debug == True:
            print('epoch', i, 'cost', J[i])
    
    print('Final cost:', J[epochs-1])
    return W, J


# In[29]:


# Now call your model with 10 neurons on hidden layer, alpha 0.05 and 10000 epochs
start = time.time()
W, J = model(X, y, nh = 10, alpha = 0.05, epochs = 10000, debug = True)
end = time.time()
print('Elapsed time', (end - start)/60, 'minutes')
print('W1 =', W['W1'])
print("b1 = ", W['b1'])
print("W2 = ", W['W2'])
print("b2 = ", W['b2'])

y_hat = predict(X, W)
acc = accuracy(y_hat, y)
print('Accuracy', acc)

plt.plot(J.keys(), J.values())
plt.title('Cost over epochs')
plt.xlabel('epochs')
plt.ylabel('cost');


# In[30]:


y_hat = predict(X, W)
acc = accuracy(y_hat, y)
print('Accuracy', acc)

plt.plot(J.keys(), J.values())
plt.title('Cost over epochs')
plt.xlabel('epochs')
plt.ylabel('cost');


# ## Time-Based Learning Rate
# 
# A problem can occur by keeping a constant learning rate during training. As our model starts to learn, it is very likely that our initial learning rate will become too big for it to continue learning. The gradient descent updates will start overshooting or circling around our minimum; as a result, the loss function will not decrease in value.
# 
# $$
# \alpha_{t+1} = \alpha * 1 / (1 + decay * epoch)
# $$
# 
# where $decay = \frac{\alpha}{epochs}$.
# 
# We will update $alpha$ every iteration.

# In[ ]:


epochs = 10000
alpha = 0.9
decay = 0.0001
alpha0 = alpha
alphs = []
for i in range(epochs):
    if i % 500 == 0:
        alpha = (alpha0 / (1 + decay * i))
        alphs.append(alpha)
plt.plot(alphs);


# In[ ]:


def model(x, y, nh, alpha = 0.1, epochs = 10000, debug = False):
    np.random.seed(2)
    nx = x.shape[1]
    ny = 1
    
    W = Xavier_init_w(nx, nh, ny)
    
    A3, Z = forward(x, W)
    print('Initial cost:', cost(A3, y))
    
    decay = 0.0001
    alpha0 = alpha
    J= {}
    for i in range(epochs):         
        alpha = (alpha0 / (1 + decay * i))
        A3, Z = forward(x, W)       
        J[i] = cost(A3, y)
        grad = backward(W, Z, X, y)
     
        #update w's and b's here
        W['W1'] = W['W1'] - alpha * grad['dW1']
        W['W2'] = W['W2'] - alpha * grad['dW2']
        W['b1'] = W['b1'] - alpha * grad['db1']
        W['b2'] = W['b2'] - alpha * grad['db2']
        
        if i % 200 == 0 and debug == True:
            print('epoch', i, 'cost', J[i])
            print('decay', decay, 'alpha', alpha)
    
    print('Final cost:', J[epochs-1])
    return W, J


# In[35]:


# Now call your model with 10 neurons on hidden layer, alpha 0.05 and 6000 epochs
start = time.time()
W, J2 = model(X, y, nh = 10, alpha = 0.05, epochs = 10000, debug=True)
end = time.time()
print('Elapsed time', (end - start)/60, 'minutes')
print('W1 =', W['W1'])
print("b1 = ", W['b1'])
print("W2 = ", W['W2'])
print("b2 = ", W['b2'])

y_hat = predict(X, W)
acc = accuracy(y_hat, y)
print('Accuracy', acc)

plt.plot(J.keys(), J.values())
plt.plot(J2.keys(), J2.values())
plt.title('Cost over epochs')
plt.xlabel('epochs')
plt.ylabel('cost');


# In[9]:


def polynomial_data(feature, degree):
    feature = np.array(feature)
    m, n = feature.shape
    x = feature
    for i in range(2,degree+1):
        tmp2 = feature**i
        x = np.concatenate([x, tmp2], axis = 1)
    
    #x1x2 = x[:,0]*x[:,1]
    #x12x2 = (x[:,0]**2)*x[:,1]
    #x1x22 = x[:,0]*(x[:,1]**2)
    #x1x2 = x1x2.reshape(-1,1)
    #x12x2 = x12x2.reshape(-1,1)
    #x1x22 = x1x22.reshape(-1,1)
    #x = np.concatenate([x, x1x2 , x12x2, x1x22], axis = 1)
    return x


# In[10]:


X = polynomial_data(X, 2)
print(X.shape)


# In[ ]:


# Now call your model with 10 neurons on hidden layer, alpha 0.05 and 6000 epochs

start = time.time()
W, J2 = model(X, y, nh = 10, alpha = 0.05, epochs = 10000, debug=True)
end = time.time()
print('Elapsed time', (end - start)/60, 'minutes')
print('W1 =', W['W1'])
print("b1 = ", W['b1'])
print("W2 = ", W['W2'])
print("b2 = ", W['b2'])

y_hat = predict(X, W)
acc = accuracy(y_hat, y)
print('Accuracy', acc)

plt.plot(J.keys(), J.values())
plt.plot(J2.keys(), J2.values())
plt.title('Cost over epochs')
plt.xlabel('epochs')
plt.ylabel('cost');


# ## Accuracy on test dataset

# In[59]:


test = pd.read_hdf('test.h5', 'df')

images = test['image'].values
y_test = test['label'].values
m = images.shape[0]

X_test = np.zeros((m, 100*100*3))
for i in range(m):
    X_test[i] = images[i].flatten().reshape(1, -1)

X_test = X_test / 255
y_test = y_test.reshape(-1, 1)
print('X_test', X_test.shape)
X_test = polynomial_data(X_test, 2)
print(X_test.shape)


# In[60]:


y_hat = predict(X_test, W)
acc = accuracy(y_hat, y_test)
print('Accuracy', acc)


# # 2. Regularization
# $$
# J(w) = \frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} \biggl( -y_{ik}\  log(\sigma_k(x_i)) - (1 - y_{ik})\ log(1 - \sigma_k(x_i)) \biggr) + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{n_{l+1}} \sum_{j=1}^{n_l} w_{lji}^2
# $$
# 
# So, derivative of the regularization term is as follows:
# 
# $$
# \frac{\partial}{\partial w_l} = \frac{\lambda}{m} w_l
# $$

# ## Backpropagation algorithm
# 
# 1. Perform feedforward pass
# 2. For each output unit $k$ in output layer $\delta_{k} = a_{k} - y_{k}$, where $y_k \in \{0, 1\}$
# 3. In hidden layers `l`, $\delta_l = \sigma'(z_l) * \delta_{l+1} w_l$
# 4. Accumulate the gradient from current example, $\Delta_l = \Delta_l + \delta_{l+1}^T a_l$
# 5. Compute the gradient for $w_l$, $\frac{\partial}{\partial w_l} = \frac{1}{m} \Delta_l + \frac{\lambda}{m} w_l$, where $\frac{\lambda}{m} w_l$ is the regularization term.

# ## Finding the optimal model

# In[39]:


def backpropagation(W, Z, x, y, lamb):

    m = x.shape[0]
    W1 = W['W1']
    W2 = W['W2']
    A2 = Z['A2']
    A3 = Z['A3']
    Z2 = Z['Z2']
    Z3 = Z['Z3']

    d3 = np.ones(y.shape)
    d2 = np.ones(Z2.shape)
    D = 0
    for j in range(m):
        d3[j] = A3[j] - y[j]
        d2[j,:] = d_sigma(Z2[j,:], 'tanh') * (d3[j] * W2)
        D = D + (d3[j] * A2[j])
    
    dW2 = (1 / m) * D + (lamb / m) * W2
        
    #d3 = (A3 - y)
    #dW2 = (1/m) * np.dot(d3.T, A2)
    db2 = (1/m) * np.sum(d3)
    #d2 = d_sigma(Z2, 'tanh') * (d3 * W2)
    dW1 = (1 /m) * np.dot(d2.T,X)
    db1 = (1 / m) * np.sum(d2,axis = 0)    
    grad = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2} 
    
    return grad

def cost_reg(W, A, y, lamb):
    W1 = W['W1'] #
    W2 = W['W2']
    #print("W1", W1.shape)
    #print("W2", W2.shape)
    m = len(y)
    reg = np.sum(np.dot(W1, W1.T) + np.dot(W2, W2.T))
    #reg = 0
    #print("reg", reg.shape)
    E = (1/m) * np.sum(-y * np.log(A) - ((1 - y) * np.log(1 - A))) + (lamb / (2 * m)) * reg
    return E


# In[40]:


def model(x, y, nh, lamb, alpha = 0.1, epochs = 1000, debug = False):
    np.random.seed(2)
    nx = x.shape[1]
    ny = 1
    m = x.shape[0]
    
    W = Xavier_init_w(nx, nh, ny)
    
    A3, Z = forward(x, W)
    print('Initial cost:', cost_reg(W, A3, y, lamb))
    
    decay = 0.0001
    alpha0 = alpha
    J= {}
    for i in range(epochs):
      
            alpha = (alpha0 / (1 + decay * i))
            A3, Z = forward(x, W)       
            J[i] = cost_reg(W, A3, y, lamb)
            grad = backpropagation(W, Z, X, y, lamb)
     
            #update w's and b's here
            W['W1'] = W['W1'] - alpha * grad['dW1']
            W['W2'] = W['W2'] - alpha * grad['dW2']
            W['b1'] = W['b1'] - alpha * grad['db1']
            W['b2'] = W['b2'] - alpha * grad['db2']
        
            if i % 200 == 0 and debug == True:
                print('epoch', i, 'cost', J[i])
                print('decay', decay, 'alpha', alpha)
    
    print('Final cost:', J[epochs-1])
    return W, J


# In[41]:


#lamb = [0]
#lamb = [0.01, 0.001, 0.0001, 0.00001]
lamb = [1, 10, 100, 1000]
print(lamb)


# In[43]:


test = pd.read_hdf('test.h5', 'df')

images = test['image'].values
y_test = test['label'].values
m = images.shape[0]

X_test = np.zeros((m, 100*100*3))
for i in range(m):
    X_test[i] = images[i].flatten().reshape(1, -1)

X_test = X_test / 255
y_test = y_test.reshape(-1, 1)
print('X_test', X_test.shape)



start = time.time()
acc = {}
acc_test = {}


for i in range(len(lamb)):
    lm = lamb[i]
    print("lambda = ", lm)
    nh = 10
    W, J = model(X, y, nh, lm, alpha = 0.05, epochs = 1000, debug=True)
    end = time.time()
    print('Elapsed time', (end - start)/60, 'minutes')
    print('W1 =', W['W1'])
    print("b1 = ", W['b1'])
    print("W2 = ", W['W2'])
    print("b2 = ", W['b2'])

    y_hat = predict(X, W)
    acc[i] = accuracy(y_hat, y)
    print('lamb = ', lamb[i],'Accuracy', acc[i])
    
    y_hat = predict(X_test, W)
    acc_test[i] = accuracy(y_hat, y_test)
    print('lamb =', lamb[i], 'Accuracy test', acc_test[i])
    plt.plot(J.keys(), J.values())

    #plt.plot(J2.keys(), J2.values())
plt.title('Cost over epochs')
plt.xlabel('epochs')
plt.ylabel('cost');


# In[ ]:
