import torch
import numpy as np


# DATA INPUT, OUTPUT
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

# CONVERT DATA TO TENSOR
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# GENERATE WEIGHT yhat = X x W + b
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# MODEL
def model(x):
    return x @ w.t() + b

# PREDICT
preds = model(inputs)

# LOSS
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

loss = mse(preds, targets)

# GRADIENT
loss.backward()

# TRAIN
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5 # gradient descent
        b -= b.grad * 1e-5 # gradient descent
        w.grad.zero_()
        b.grad.zero_()

# PREDICT
preds = model(inputs)
loss = mse(preds, targets)



