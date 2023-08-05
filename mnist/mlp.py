import numpy as np
import Relu
import softmax_with_loss
from collections import OrderedDict
import linear
class MLP:
    def __init__(self, layers_size):      
        self.layer = len(layers_size)-2 
        self.params = {} 
        self.params['w1'] = (np.random.randn(layers_size[1], layers_size[0]) * np.sqrt(2.0 / layers_size[0])).T 
        self.params['b1'] = np.zeros(layers_size[1])  
        for i in range(self.layer):
            self.params['w'+str(i+2)] = (np.random.randn(layers_size[i+2], layers_size[i+1]) * np.sqrt(2.0 / layers_size[i+1])).T 
            self.params['b'+str(i+2)] = np.zeros(layers_size[i+2])      
        self.layers = OrderedDict()
        self.layers['Linear1'] = linear.Linear(self.params['w1'], self.params['b1'])
        for i in range(self.layer):
            self.layers['Relu'+str(i+1)] = Relu.Relu()
            self.layers['Linear'+str(i+2)] = linear.Linear(self.params['w'+str(i+2)], self.params['b'+str(i+2)])     
        self.output = softmax_with_loss.SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)  
        return x  
    def loss(self, x, t):
        y = self.predict(x)
        return self.output.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1) 
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy       
    def gradient(self, x, t):
        self.loss(x, t) 
        dout = self.output.backward() 
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers: 
            dout = layer.backward(dout)
        grads = {}
        grads['w1'], grads['b1'] = self.layers['Linear1'].dw, self.layers['Linear1'].db
        for i in range(self.layer):
            grads['w'+str(i+2)], grads['b'+str(i+2)] = self.layers['Linear'+str(i+2)].dw, self.layers['Linear'+str(i+2)].db
        return grads