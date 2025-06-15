from tensor import Tensor
import numpy as np

class Module: # Every lego piece fits like this, must have a set of parameters, able to zer_grad and a calling mechanism
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers: # Iterate the input through each layer
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
class Linear(Module): # Linear layer
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.01)
        self.b = Tensor(np.zeros(out_features))
        
    def forward(self, x: Tensor):
        out = x @ self.W
        out = out + self.b
        return out
    
    def parameters(self):
        return [self.W, self.b]