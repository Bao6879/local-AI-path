import random
from value import Value
class Neuron():
    def __init__(self, inputs):
        self.weights=[Value(random.uniform(-1, 1)) for i in range(inputs)]
        self.bias=Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        tmp=sum(w*t for w, t in zip(self.weights, x))
        ret=tmp+self.bias
        return ret.tanh()
    
    def params(self):
        return self.weights+[self.bias]

class Layer():
    def __init__(self, inputs, output):
        self.neurons=[Neuron(inputs) for i in range(output)]
    
    def __call__(self, x):
        output=[n(x) for n in self.neurons]
        if len(output)==1:
            return output[0]
        return output
    
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

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MultiLayer(3, [20, 20, 1])
for k in range(100):
  
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  n.zeroGrad()
  loss.backward()
  
  for p in n.params():
    p.data += -0.1 * p.grad
  
  print(k, loss.data)