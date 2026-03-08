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
blockSize=3
features=10
random.seed(3108)
trainSplitSize=0.8
hiddenLayerNeurons=200
learningRates=[(0, 0.1), (50000, 0.05), (100000, 0.01), (1e9, 0.001)] #Learning rate decay
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
g=torch.Generator().manual_seed(3108)
C=torch.randn((27, features), generator=g)
w1=torch.randn((blockSize*features, hiddenLayerNeurons), generator=g) #Hidden layer
b1=torch.randn((hiddenLayerNeurons), generator=g)
w2=torch.randn((hiddenLayerNeurons, 27), generator=g)
b2=torch.randn((27), generator=g)

parameters=[C, w1, b1, w2, b2]
for p in parameters:
    p.requires_grad=True

for i in range(10000):
    #Batch
    ix=torch.randint(0, trX.shape[0], (batchSize, ))

    #Forward pass
    emb=C[trX[ix]]
    h=torch.tanh(emb.view(batchSize, blockSize*features)@w1+b1) #Output of hidden layer
    end=h@w2+b2 #End predictions
    loss=F.cross_entropy(end, trY[ix]) 

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
    if i%100==0:
        print(i, loss.data.item())

#Final training loss
emb=C[trX]
h=torch.tanh(emb.view(len(trX), blockSize*features)@w1+b1) #Output of hidden layer
end=h@w2+b2 #End predictions
trLoss=F.cross_entropy(end, trY) 
print(f'Training loss: {trLoss.data}')

#Test loss
emb=C[tsX]
h=torch.tanh(emb.view(len(tsX), blockSize*features)@w1+b1) #Output of hidden layer
end=h@w2+b2 #End predictions
tsLoss=F.cross_entropy(end, tsY) 
print(f'Test loss: {tsLoss.data}')
#Running time: 