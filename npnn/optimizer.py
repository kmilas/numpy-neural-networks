import numpy as np

class SGD(object):
    def __init__(self, model, lr, momentum):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.momentum_dW = []
        self.momentum_db = []

        for layer in model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                self.momentum_dW.append(np.zeros(layer.dW.shape))
                self.momentum_db.append(np.zeros(layer.db.shape))
    
    def step(self):
        i = 0
        for layer in self.model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                self.momentum_dW[i] = self.momentum*self.momentum_dW[i] + layer.dW
                self.momentum_db[i] = self.momentum*self.momentum_db[i] + layer.db
                layer.W -= self.lr*self.momentum_dW[i]
                layer.b -= self.lr*self.momentum_db[i]
                i+=1


    def zero_grad(self):
        for layer in self.model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                layer.dW.fill(0.0)
                layer.db.fill(0.0)
