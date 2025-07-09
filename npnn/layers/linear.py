import numpy as np

class Linear(object):
    def __init__(self,in_neurons, out_neurons):
        # ,init_fn=kaiming_init
        self.W = np.random.randn(in_neurons, out_neurons)
        self.b = np.zeros((1,out_neurons))
    
        self.dW = np.zeros((in_neurons, out_neurons))
        self.db = np.zeros((1,out_neurons))
        self.cache = None

        self.W = self.kaiming_init_weights()
        self.b = self.kaiming_init_biases()

    def kaiming_init_weights(self):
        std = np.sqrt(2.0) / np.sqrt(self.W.shape[0]) # fan_in mode
        bounds = np.sqrt(3.0) * std
        return np.random.uniform(-bounds, bounds, self.W.shape)
    
    def kaiming_init_biases(self):
        bounds = 1. / np.sqrt(self.W.shape[0])
        return np.random.uniform(-bounds, bounds, self.b.shape)

    def count_params(self):
        return self.W.shape[0]*self.W.shape[1] + self.b.shape[1]

    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        self.cache = x

        return x@self.W + self.b
    
    def backward(self,delta):
        batch_size = self.cache.shape[0]

        dX = (delta@self.W.T)
        self.dW = (self.cache.T @ delta)/ batch_size
        self.db = np.mean(delta, axis=0, keepdims=True)

        return dX