import numpy as np

class ReLU(object):
    def __init__(self):
        self.cache = None

    def forward(self,x):
        activation = np.maximum(0,x)
        self.cache = activation 
        
        return activation
    
    def __call__(self,x):
        return self.forward(x)

    def backward(self, dX):
        
        return ((self.cache > 0)*1)*dX

class Sigmoid(object):
    def __init__(self):
        self.cache = None

    def forward(self,x):
        self.cache = (np.exp(x)/ (1 + np.exp(x)))
        return self.cache
    
    def __call__(self,x):
        return self.forward(x)

    def backward(self, dX):
        return self.cache*(1 - self.cache)*dX