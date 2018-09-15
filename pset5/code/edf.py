### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf

# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad

## Fill this out        
def init_momentum():
    # initialize another grad to save last grad
    for p in params:
        p.grad_1 = np.zeros_like(p.top)

## Fill this out
def momentum(lr,mom=0.9):
    for p in params:
        p.grad_1 = p.grad + mom * p.grad_1 # p.grad_1 in the right is old (gi), p.grad_1 in the left is new
        p.top = p.top - lr*p.grad_1

###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)  #(50, 3136) (2304, 10)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)

# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)

# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))


# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])  # (50, 3136)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)
            
# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):     # x is 4-D array (B * H * W * 1)
        ops.append(self)
        self.x = x
        self.k = k

    def forward(self):          #compute convolution (save in self.top, initial 64 output channels)
        # Does Kernal need to be flipped?
        self.k.top = np.flip(self.k.top, axis = 0)
        self.k.top = np.flip(self.k.top, axis = 1)

        self.top = np.zeros((self.x.top.shape[0], self.x.top.shape[1] - self.k.top.shape[0] + 1, self.x.top.shape[2] - self.k.top.shape[1] + 1,
                             self.k.top.shape[3]))  # (50, 25, 25, 8) or (50, 12, 12, 16)

        for batch in range(self.x.top.shape[0]):
            for C2 in range(self.k.top.shape[3]):
                sum = np.zeros((self.top.shape[1], self.top.shape[2]))
                for C1 in range(self.k.top.shape[2]):
                    sum = sum + convolve2d(self.x.top[batch, :, :, C1], self.k.top[:, :, C1, C2], "valid")
                self.top[batch, :, :, C2] = sum

    def backward(self):
        if self.x in ops or self.x in params:       # to the input images
            for batch in range(self.x.grad.shape[0]):
                for C1 in range(self.x.grad.shape[3]):
                    sum = np.zeros((self.x.grad.shape[1], self.x.grad.shape[2]))
                    for C2 in range(self.k.top.shape[3]):
                        sum = sum + convolve2d(self.grad[batch, :, :, C2], self.k.top[:, :, C1, C2] * self.k.grad[:, :, C1, C2], 'full')
                    self.x.grad[batch, :, :, C1] = self.x.grad[batch, :, :, C1] + sum
        
        if self.k in ops or self.k in params:       # to the kernel
            for C1 in range(self.k.grad.shape[2]):
                for C2 in range(self.k.grad.shape[3]):
                    for batch in range(self.grad.shape[0]):
                        self.k.grad[:, :, C1, C2] = self.k.grad[:, :, C1, C2] + \
                                                  convolve2d(self.x.top[batch, :, :, C1], np.rot90(self.grad[batch, :, :, C2], 2), 'valid')
        
        #self.grad is conv2 output's gradients

#kernal scan the matrix with stride step
class conv2WithStride:
    def __init__(self, x, k, stride):  # x is 4-D array (B * H * W * 1)
        ops.append(self)
        self.x = x
        self.k = k
        self.stride = stride

    def forward(self):  # compute convolution (save in self.top, initial 64 output channels)
        # Does Kernal need to be flipped?
        self.k.top = np.flip(self.k.top, axis=0)
        self.k.top = np.flip(self.k.top, axis=1)

        self.top = np.zeros((self.x.top.shape[0], (self.x.top.shape[1] - self.k.top.shape[0]) + 1,
                             (self.x.top.shape[2] - self.k.top.shape[1]) + 1,
                             self.k.top.shape[3]))  # (50, 13, 13, 8) or (50, 6, 6, 16)

        for batch in range(self.x.top.shape[0]):
            for C2 in range(self.k.top.shape[3]):
                sum = np.zeros((self.top.shape[1], self.top.shape[2]))  #(13,13)
                for C1 in range(self.k.top.shape[2]):
                    sum = sum + convolve2d(self.x.top[batch, :, :, C1],
                                            self.k.top[:, :, C1, C2], "valid")
                self.top[batch, :, :, C2] = sum

    def backward(self):
        if self.x in ops or self.x in params:  # to the input images
            for batch in range(self.x.grad.shape[0]):
                for C1 in range(self.x.grad.shape[3]):
                    sum = np.zeros((self.x.grad.shape[1], self.x.grad.shape[2]))
                    for C2 in range(self.k.top.shape[3]):
                        sum = sum + convolve2d(self.grad[batch, :, :, C2],
                                               self.k.top[:, :, C1, C2] * self.k.grad[:, :, C1, C2], 'full')
                    self.x.grad[batch, :, :, C1] = self.x.grad[batch, :, :, C1] + sum

        if self.k in ops or self.k in params:  # to the kernel
            for C1 in range(self.k.grad.shape[2]):
                for C2 in range(self.k.grad.shape[3]):
                    for batch in range(self.grad.shape[0]):
                        self.k.grad[:, :, C1, C2] = self.k.grad[:, :, C1, C2] + \
                                                    convolve2d(self.x.top[batch, :, :, C1],
                                                               np.rot90(self.grad[batch, :, :, C2], 2), 'valid')