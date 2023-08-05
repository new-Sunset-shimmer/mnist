import numpy as np
class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.x_shape = None
        self.dw = None
        self.db = None
    def forward(self, x):
        self.x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.x_shape)  
        return dx