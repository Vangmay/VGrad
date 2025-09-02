# VGrad

VGrad is a minimal deep learning framework implemented in Python and NumPy. It provides core components for building and training neural networks, including tensors, modules, optimizers, and loss functions.

---

## File Overview

### `tensor.py`

Defines the `Tensor` class, the fundamental data structure for VGrad. It wraps NumPy arrays and supports automatic differentiation via a simple computational graph.

- Basic arithmetic operations (`+`, `-`, `*`, `@`, `**`)
- Backpropagation (`backward`)
- Gradient storage (`grad`)
- Zeroing gradients (`zero_grad`)
- Handles graph construction for autograd

### `modules.py`

Contains the base class for all neural network modules and layers.

- `Module`: Abstract base class for all layers and models. Handles parameter management and gradient zeroing.
- `Sequential`: Container for stacking layers sequentially.
- `Linear`: Implements a fully connected (dense) layer with weights and bias.

### `losses.py`

Implements loss functions for training.

- `MSELoss`: Mean Squared Error loss, computes the squared difference between predictions and targets.

### `optim.py`

Provides optimization algorithms for updating model parameters.

- `SGD`: Stochastic Gradient Descent optimizer. Updates parameters using their gradients and a learning rate.

### `activations.py`

Reserved for activation functions (currently empty). Typical activations like ReLU, Sigmoid, etc., can be implemented here.

### `utils.py`

Reserved for utility functions (currently empty). Helper functions for data processing, initialization, etc., can be added here.

### `VGrad.ipynb`

Jupyter notebook for interactive experimentation, model building, and training using VGrad components.

---

## Usage Example

```python
from modules import Sequential, Linear
from tensor import Tensor
from losses import MSELoss
from optim import SGD

# Define a model
model = Sequential([Linear(2, 4), Linear(4, 1)])

# Forward pass
x = Tensor([[1.0, 2.0]])
output = model(x)

target = Tensor([[3.0]])
loss_fn = MSELoss()
loss = loss_fn(output, target)

# Backward pass
loss.backward()

# Optimizer step
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()
```

For more details, refer to the code in each file. VGrad is designed for educational purposes and rapid prototyping.
