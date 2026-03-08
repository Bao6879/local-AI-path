import random
import torch
import torch.nn.functional as F

names=open('names.txt', 'r').read().splitlines()

iToC={0: '.'}
idx=0
for i in range(97, 123):
    iToC[chr(i)]=idx
    idx+=1
cToI={s:i for i, s in iToC}

#Settings:
blockSize=3
features=10
random.seed(3108)
trainSplitSize=0.8

#Set up
#Data
random.shuffle(names)
trX, trY=[], []
tsX, tsY=[], []
idx=0
for w in names[:5]:

    ctx=[0]*blockSize
    for ch in w+'.':
        if idx<=len(names)*trainSplitSize:
            trX.append(ctx)
            trY.append(cToI[ch])
        else:
            tsX.append(ctx)
            tsY.append(cToI[ch])
        ctx=ctx[1:]+ch
    
    idx+=1
    
trX=torch.tensor(trX)
trY=torch.tensor(trY)
tsX=torch.tensor(tsX)
tsY=torch.tensor(tsY)