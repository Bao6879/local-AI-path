import torch
from Engine.value import Value

def simpleTest():
    x=Value(4.0)
    y=Value(3.0)
    z=Value(1.0)
    t=Value(-5.0)
    f=(x*y+t)*(z-t)-(x+y)*t+pow(z, 2)*y+x*3-5*t
    f.backward()
    endTest=f
    startTest=[x.grad, y.grad, z.grad, t.grad]

    x=torch.tensor([4.0]).double()
    y=torch.tensor([3.0]).double()
    z=torch.tensor([1.0]).double()
    t=torch.tensor([-5.0]).double()
    x.requires_grad=y.requires_grad=z.requires_grad=t.requires_grad=True
    f=(x*y+t)*(z-t)-(x+y)*t+pow(z, 2)*y+x*3-5*t
    f.backward()
    endTrue=f
    startTrue=[x.grad.item(), y.grad.item(), z.grad.item(), t.grad.item()]

    print(endTest.data==endTrue.data.item())
    print(startTest)
    print(startTrue)
    #Not equal :(

def simplerTest():
    pass

simpleTest()