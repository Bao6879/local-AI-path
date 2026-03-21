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
numHeads=8 #Num heads*head size = feature length.
headSize=16
maxIter=5000 #Number of epochs to run
evalInterval=100 #Every interval, run full evaluation
evalIters=100 #Iterations in evaluation
learningRate=1e-3
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
    def getCurrentLoss():
        out={}
        model.eval() #Evaluation mode
        for split in ['train', 'val']:
            losses = torch.zeros(evalIters)
            for k in range(evalIters):
                X, Y = getBatch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train() #Back to training mode
        return out

class FeedForward(nn.Module): #Simple MLP for thinking on the data
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(featuresLength, featuresLength*4), 
            nn.ReLU(), 
            nn.Linear(featuresLength*4, featuresLength))
    def forward(self, x):
        return self.net(x)

class MultiHead(nn.Module): #Concat the results from multiple attention heads
    def __init__(self):
        super().__init__()
        self.heads=nn.ModuleList([Head(headSize) for _ in range(numHeads)]) #Get multiple heads
        self.proj=nn.Linear(featuresLength, featuresLength)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1)
        out=self.proj(out)
        return out;

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

class Block(nn.Module): #A transformer block
    def __init__(self):
        super().__init__()
        self.sa=MultiHead(numHeads, headSize)
        self.ffwd=FeedForward(featuresLength)
    def forward(self, x):
        x=x+self.sa(x)
        x=x+self.ffwd(x)
        return x

class Model(nn.Module):
    def __init__(self):
        self.tokenEmbeddingTable=nn.Embedding(vocabSize, featuresLength) #Each character has features
        self.positionEmbeddingTable=nn.Embedding(contextLength, featuresLength) #Each position has features.
        self.blocks=nn.Sequential(Block(), Block(), Block())
        self.lm_head=nn.Linear(featuresLength, vocabSize) #Turns the embeddings (features) into logits of vocab size
    
    def forward(self, idx, targets=None):
        B, T=idx.shape

        tokenEmbed=self.tokenEmbeddingTable(idx) #Get the embeddings from the input tokens
        positionEmbed=self.positionEmbeddingTable(torch.arange(T, device=device)) #T*C, features for each position
        x=tokenEmbed+positionEmbed #B, T, C
        x=self.blocks(x)
        logits=self.lm_head(tokenEmbed) #Convert them to logits

        if targets is None:
            loss=None
        else:
            B, T, C=logits.shape #Batch, context length, feature length
            logits=logits.view(B*T, C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, tokenCount):
        #Generate token count of tokens
        for _ in range(tokenCount):
            idxCropped=idx[:, -contextLength:]
            #Get prediction
            logits, loss=self(idxCropped)
            #Get the last position
            logits=logits[:, -1, :]
            probs=F.softmax(logits, dim=-1)
            #Get prediction
            next=torch.multinomial(probs, num_samples=1)
            #Append to the end
            idx=torch.cat((idx, next), dim=1)
        return idx


model=Model()
model=model.to(device)
optimizer=torch.optim.AdamW(model.parameters(), lr=learningRate)

for iter in range(maxIter):
    if iter%evalInterval==0:
        losses=getCurrentLoss()
        print(f'Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    #Get a batch 
    xb, yb=getBatch('train')

    #Training
    logits, loss=model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))