import numpy as np
from softax import softmax
from loss_func import log_like
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None    
        self.t = None    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = log_like(self.y, self.t)   
        return self.loss
    def backward(self):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size