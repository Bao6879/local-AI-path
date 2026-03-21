import torch
import torch.nn as nn
from torch.nn import functional as F

#Setup
with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read() #Everything in the file.
chars=list(set(text)) #All unique characters in the files.
vocabSize=len(chars)

cToI={c:i for i, c in enumerate(chars)} #Character -> index and vice versa
iToC={i:c for i, c in enumerate(chars)}
encode=lambda s: [cToI[c] for c in s] #Takes in a string -> turns to array of ints and vice versa

decode=lambda a: ''.join([iToC[i] for i in a]) 
data=torch.tensor(encode(text), dtype=torch.int) #Encoded dataset, stored in a tensor
tmp=int(0.9*len(data))
trainData=data[:tmp] #Split of 9:1 train:test
testData=data[tmp:]

#Settings
batchSize=4 #At once, in parallel
contextLength=8 #Up to this many characters for predictions
featuresLength=12 #Features for each character
headSize=16
seed=3108

device='cuda' if torch.cuda.is_available() else 'cpu' #Use GPU to drastically accelerate process if possible
torch.manual_seed(seed=seed)

def getBatch(split):
    data=trainData if split=='train' else testData
    ix=torch.randint(len(data)-contextLength, (batchSize, ))
    x=torch.stack([data[i:i+contextLength] for i in ix])
    y=torch.stack([data[i+1:i+contextLength+1] for i in ix])
    x, y=x.to(device), y.to(device)
    return x, y

with torch.no_grad():
    def finalLoss():
        out={}

class Head(nn.Module): #Head of self attention
    def __init__(self):
        super().__init__()
        self.key=nn.Linear(featuresLength, headSize, bias=False) #What I have to share
        self.query=nn.Linear(featuresLength, headSize, bias=False) #What I want to know
        self.value=nn.Linear(featuresLength, headSize, bias=False) #What I will share
        self.register_buffer('tril', torch.tril(torch.ones(contextLength, contextLength)))
    
    def forward(self, x):
        B, T, C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        #How much do I want to know about each position?
        w=q@k.transpose(-2, -1)*C**-0.5
        w=w.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        w=F.softmax(w, dim=-1)
        #Get the info
        out=w@v
        return out

class Model(nn.Module):
    def __init__(self):
        self.tokenEmbeddingTable=nn.Embedding(vocabSize, featuresLength) #Each character has features
        self.positionEmbeddingTable=nn.Embedding(contextLength, featuresLength) #Each position has features.
        self.lm_head=nn.Linear(featuresLength, vocabSize) #Turns the embeddings (features) into logits of vocab size
    
    def forward(self, idx, targets=None):
        B, T=idx.shape

        tokenEmbed=self.tokenEmbeddingTable(idx) #Get the embeddings from the input tokens
        positionEmbed=self.positionEmbeddingTable(torch.arange(T, device=device)) #T*C, features for each position
        x=tokenEmbed+positionEmbed #B, T, C
        logits=self.lm_head(tokenEmbed) #Convert them to logits

        if targets is None:
            loss=None
        else:
            B, T, C=logits.shape #Batch, context length, feature length
            logits=logits.view(B*T, C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self):
        pass