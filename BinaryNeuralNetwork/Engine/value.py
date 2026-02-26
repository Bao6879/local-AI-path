import math
class Value:
    def __init__(self, data, children=()):
        self.data=data 
        self.grad=0 #how much changing this would affect the final result
        self._backward=lambda:None
        self.prev=set(children)
    
    def __repr__(self):
        return "Value("+str(self.data)+"), grad("+str(self.grad)+")"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other=Value(other)
        new=Value(self.data+other.data, (self, other))
        
        def _backward():
            self.grad+=new.grad
            other.grad+=new.grad
        new._backward=_backward
        return new

    def __mul__(self, other):
        if not isinstance(other, Value):
            other=Value(other)
        new=Value(self.data*other.data, (self, other))
        
        def _backward():
            self.grad+=other.data*new.grad
            other.grad+=self.data*new.grad
        new._backward=_backward
        return new
    
    def __pow__(self, other):
        new=Value(self.data**other, (self,))
        
        def _backward():
            self.grad+=other*self.data**(other-1)*new.grad
        new._backward=_backward
        return new

    def tanh(self):
        new=Value((pow(math.e, self.data*2)-1)/(pow(math.e, self.data*2)+1), (self,))

        def _backward():
            self.grad+=(1-new.data*new.data)*new.grad
        new._backward=_backward
        return new
    
    def __sub__(self, other):
        return self+(-other)
    
    def __neg__(self):
        return Value(-self.data)
    
    def __radd__(self, other):
        return self+other
    
    def __rmul__(self, other):
        return self*other

    def backward(self):
        # topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go backwards 1 at a time to find grad
        self.grad = 1
        for v in reversed(topo):
            v._backward()