import torch
import math
from Engine.value import Value

#Easy test, 1 variable, basic arithmetics
def easyTest():
    x=Value(-3.0)
    f=4-x/5+6*x-x-3+4*x/6
    f.backward()
    endTest=f
    startTest=[x.grad]

    x=torch.tensor([-3.0]).double()
    x.requires_grad=True
    f=4-x/5+6*x-x-3+4*x/6
    f.backward()
    endTrue=f
    startTrue=[x.grad]

    return startTest==startTrue and endTest.data==endTrue.data.item()

#Medium test, 4 variables, basic arithmetics
def mediumTest():
    x=Value(4.0)
    y=Value(3.0)
    z=Value(1.0)
    t=Value(-5.0)
    f=(x*y+t)/(z-t)-(x+y)*t+pow(z, 2)*y+x*3-5*t
    f.backward()
    endTest=f
    startTest=[x.grad, y.grad, z.grad, t.grad]

    x=torch.tensor([4.0]).double()
    y=torch.tensor([3.0]).double()
    z=torch.tensor([1.0]).double()
    t=torch.tensor([-5.0]).double()
    x.requires_grad=y.requires_grad=z.requires_grad=t.requires_grad=True
    f=(x*y+t)/(z-t)-(x+y)*t+pow(z, 2)*y+x*3-5*t
    f.backward()
    endTrue=f
    startTrue=[x.grad.item(), y.grad.item(), z.grad.item(), t.grad.item()]

    return startTest==startTrue and endTest.data==endTrue.data.item()

#Hard test, 6 vars, more complicated math
def hardTest():
    x=Value(4.1)
    y=Value(3.8)
    z=Value(1.3)
    a=Value(-5.6)
    b=Value(2.5)
    c=Value(10.5)
    f=((x*y-a)*(b*(c-z/(a+c))))-4/a+5*pow(a, 6)/pow(z, 4)-3*a*3*(x-a*y+b/c*z)
    f.backward()
    endTest=f
    startTest=[x.grad, y.grad, z.grad, a.grad, b.grad, c.grad]
    endTest=Value(math.floor(endTest.data*100)/100)

    x=torch.tensor([4.1]).double()
    y=torch.tensor([3.8]).double()
    z=torch.tensor([1.3]).double()
    a=torch.tensor([-5.6]).double()
    b=torch.tensor([2.5]).double()
    c=torch.tensor([10.5]).double()
    x.requires_grad=y.requires_grad=z.requires_grad=a.requires_grad=b.requires_grad=c.requires_grad=True
    f=((x*y-a)*(b*(c-z/(a+c))))-4/a+5*pow(a, 6)/pow(z, 4)-3*a*3*(x-a*y+b/c*z)
    f.backward()
    endTrue=f
    startTrue=[x.grad.item(), y.grad.item(), z.grad.item(), a.grad.item(), b.grad.item(), c.grad.item()]
    endTrue=math.floor(endTrue.data.item()*100)/100


    difference=[abs(test-true) for test, true in zip(startTest, startTrue)]
    return [di<5e-1 for di in difference] and endTest.data==endTrue
    #Adjustment for rounding errors

#Debugging
def debugTest():
    x=Value(2)
    f=10-x
    f.backward()
    endTest=f
    startTest=x.grad
               
    x=torch.tensor([2]).double()
    x.requires_grad=True
    f=10-x
    f.backward()
    endTrue=f
    startTrue=x.grad.item()

    return startTest==startTrue and endTest.data==endTrue.data.item()

def allTests():
    if easyTest()==mediumTest()==debugTest()==hardTest()==True:
        print("All tests passed")
    else:
        print("Something's wrong")

allTests()