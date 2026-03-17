import random
import torch
import torch.nn.functional as F

names=open('names.txt', 'r').read().splitlines()

cToI={'.': 0}
idx=0
for i in range(97, 123):
    cToI[chr(i)]=idx
    idx+=1
iToC={i:s for s,i in cToI.items()}

#Settings:
blockSize=8
features=10
seed=3108
random.seed(seed)
trainSplitSize=0.8
hiddenLayerNeurons=200
learningRates=[(0, 0.1), (25000, 0.05), (50000, 0.01), (100000, 0.001), (1e9, 0.001)] #Learning rate decay
batchSize=64

#Set up
#Data
random.shuffle(names)
trX, trY=[], []
tsX, tsY=[], []
idx=0
for w in names:

    ctx=[0]*blockSize
    for ch in w+'.':
        tmp=cToI[ch]
        if idx<=len(names)*trainSplitSize:
            trX.append(ctx)
            trY.append(tmp)
        else:
            tsX.append(ctx)
            tsY.append(tmp)
        ctx=ctx[1:]+[tmp]
    
    idx+=1
    
trX=torch.tensor(trX)
trY=torch.tensor(trY)
tsX=torch.tensor(tsX)
tsY=torch.tensor(tsY)

#Neural net
g=torch.Generator().manual_seed(seed)
C=torch.randn((27, features), generator=g)
w1=torch.randn((blockSize*features, hiddenLayerNeurons), generator=g)*0.1 #Hidden layer
#b1=torch.randn((hiddenLayerNeurons), generator=g)*0.01
w2=torch.randn((hiddenLayerNeurons, 27), generator=g)*0.01
#b2=torch.randn((27), generator=g)*0

bngain=torch.ones((1, hiddenLayerNeurons))
bnbias=torch.zeros((1, hiddenLayerNeurons))

parameters=[C, w1, w2, bngain, bnbias]
for p in parameters:
    p.requires_grad=True

#Torchifying NN
class Linear:
    def __init__(self, nin, nout, bias=True):
        self.weight=torch.randn((nin, nout), generator=g)
        self.bias=torch.zeros(nout) if bias else None
    def __call__(self, x):
        self.out=x@self.weight
        if self.bias is not None:
            self.out+=self.bias
        return self.out
    def params(self):
        return [self.weight]+([] if self.self.bias is None else [self.bias])

class BNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps=eps
        self.momentum=momentum
        self.training=True

        self.gamma=torch.ones(dim)
        self.beta=torch.zeros(dim)

        self.runMean=torch.zeros(dim)
        self.runVar=torch.ones(dim)
    def __call__(self, x):
        if self.training:
            xmean=x.mean(0, keepdim=True)
            xvar=x.var(0, keepdim=True)
        else:
            xmean=self.runMean
            xvar=self.runVar
        x=(x-xmean)/torch.sqrt(xvar+self.eps)
        self.out=self.gamma*x+self.beta
        if self.training:
            with torch.no_grad():
                self.runMean=(1-self.momentum)*self.runMean+self.momentum*xmean
                self.runVar=(1-self.momentum)*self.runVar+self.momentum*xvar
        return self.out
    def params(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out=torch.tanh(x)
        return self.out
    def params(self):
        return []

def flatten(x):
    B, T, C=x.shape
    x=x.view(B, T//2, C*2)
    if x.shape[1]==1:
        x=x.squeeze(1)
    return x

def forwardPass(emb):
    h=flatten(emb) #Flatten
    h=h@w1 #Linear layer
    h=torch.tanh(bngain*(h-h.mean(0, keepdim=True))/(h.std(0, keepdim=True))+bnbias) #batch normalization + tanh

    #Layer 2:
    h=flatten(h)
    h=h@w2
    h=torch.tanh(bngain*(h-h.mean(0, keepdim=True))/(h.std(0, keepdim=True))+bnbias)

    #Layer 3:
    h=flatten(h)
    h=h@w2
    h=torch.tanh(bngain*(h-h.mean(0, keepdim=True))/(h.std(0, keepdim=True))+bnbias)
    return h

for i in range(200000):
    #Batch
    ix=torch.randint(0, trX.shape[0], (batchSize, ))

    #Forward pass
    #Layer 1:
    emb=C[trX[ix]]
    h=forwardPass(emb)

    loss=F.cross_entropy(h, trY[ix]) 
    #Backward pass
    for p in parameters:
        p.grad=None
    loss.backward()

    for r in learningRates:
        if i<r[0]:
            break
        lr=r[1]

    for p in parameters:
        p.data+=-lr*p.grad
    if (i+1)%10000==0:
        print(i, loss.data.item())

#Calibrate the batch norm at the end
with torch.no_grad():
    emb=C[trX]
    h=emb.view(emb.shape[0], blockSize*features)@w1
    bnmean=h.mean(0, keepdim=True)
    bnstd=h.std(0, keepdim=True)


#Final training loss
emb=C[trX]
h=forwardPass(emb)
trLoss=F.cross_entropy(h, trY) 
print(f'Training loss: {trLoss.data}')

#Test loss
emb=C[tsX]
h=forwardPass(emb)
tsLoss=F.cross_entropy(h, tsY) 
print(f'Test loss: {tsLoss.data}')

#Sampling
g=torch.Generator().manual_seed(10)
for _ in range(20):
    s=""
    ctx=[0]*blockSize
    while True:
        emb=C[torch.tensor(ctx)]
        h=forwardPass(emb)
        probs=F.softmax(h, dim=1)
        ix=torch.multinomial(probs, num_samples=1, generator=g).item()
        ctx=ctx[1:]+[ix]
        s+=iToC[ix]
        if ix==0:
            break
    
    print(s)