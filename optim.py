class SGD:
    def __init__(self, parameters, lr = 0.01):
        self.parameters = parameters
        self.lr = lr
        
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad.reshape(p.data.shape)
    
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
