## Engine.py

import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)


        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for graph debugging

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children = (self, other), _op="+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data, (self,), _op='neg')

        def _backward():
            self.grad += -1 * out.grad

        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
        
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children = (self, other), _op = "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            grad = np.ones_like(self.data) * out.grad
            self.grad += grad 

        out._backward = _backward
        return out
    
    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,), f'**{power}')
        
        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = 0
