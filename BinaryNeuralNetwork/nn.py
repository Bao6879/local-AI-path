import random
from value import Value
class Neuron():
    def __init__(self, inputs):
        self.weights=[Value(random.uniform(-1, 1)) for i in range(inputs)]
        self.bias=Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        sum=0
        for i in range(len(self.weights)):
            sum+=self.weights[i]*x[i]
        sum+=self.bias
        return sum.tanh()
    
    def params(self):
        return self.weights+[self.bias]

class Layer():
    def __init__(self, inputs, output):
        self.neurons=[Neuron(inputs) for i in range(output)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def params(self):
        return [p for i in self.neurons for p in i.params()]

class MultiLayer():
    def __init__(self, inputs, layersSizes):
        tmp=[inputs]+layersSizes
        self.layers=[Layer(tmp[i], tmp[i+1]) for i in range(len(tmp)-1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def params(self):
        return [p for i in self.layers for p in i.params()]
    
    def zeroGrad(self):
        for p in self.params():
            p.grad=0