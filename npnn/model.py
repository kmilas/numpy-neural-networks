from .layers.activation import ReLU
from .layers.linear import Linear

class ModuleList(object):
    def __init__(self):
        self.layers = []
    
    def forward(self, x):
        for layer in self.layers:
           x = layer(x)
        
        return x
    
    def __call__(self,x):
        return self.forward(x)

    
    def backward(self, dX):
        for layer in reversed(self.layers):
           dX = layer.backward(dX)

class FFN(ModuleList):
    """ 
    Feed Forward Network
    """
    def __init__(self, input_size, out_size, hidden_layers, activation):
        self.layers = []
        self.neurons = [input_size, *hidden_layers]

        for i in range(len(self.neurons)-1):
            self.layers.append(Linear(self.neurons[i], self.neurons[i+1]))
            self.layers.append(activation())

        # Last layer without activation
        self.layers.append(Linear(self.neurons[-1], out_size))

    def count_params(self):
        params = 0.0
        for layer in self.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                params += layer.count_params()
        return params
