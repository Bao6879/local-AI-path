import torch
import time
import math
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

#Setup
with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read() #Everything in the file.
enc=tiktoken.get_encoding('gpt2')
tmpTokens=enc.encode(text)
finalTokens=torch.tensor(tmpTokens)

tmp=int(0.9*len(finalTokens)) #The encoded dataset, stored in a tensor
trainData=finalTokens[:tmp] #Split of 9:1 train:test
testData=finalTokens[tmp:]

#Settings
vocabSize=50304
numLayers=23
dropout=0.2
batchSize=4 #16, At once, in parallel
contextLength=128 #1024, Up to this many characters for predictions
featuresLength=384 #768, Features for each character
numHeads=6 #12, Num heads*head size = feature length.
headSize=64
maxIter=10 #Number of epochs to run
evalInterval=500 #Every interval, run full evaluation
evalIters=200 #Iterations in evaluation
learningRate=6e-4
seed=3108

device='cuda' if torch.cuda.is_available() else 'cpu' #Use GPU to accelerate process if possible
torch.manual_seed(seed=seed)

#Cosine learning rate decay with warmup (like GPT-3)
minLr=learningRate*0.1
warmUp=maxIter*0.2
def getLr(i):
    if i<warmUp:
        return learningRate*(i+1)/warmUp
    decayRate=(i-warmUp)/(maxIter-warmUp)
    coeff=0.5*(1.0+math.cos(math.pi*decayRate))
    return minLr+coeff*(learningRate-minLr)

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
        projLayer=nn.Linear(featuresLength*4, featuresLength)
        projLayer.stdDiffer=True
        self.net=nn.Sequential(
            nn.Linear(featuresLength, featuresLength*4), 
            nn.GELU(approximate='tanh'), 
            projLayer,
            nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.net(x)

class MultiHead(nn.Module): #Concat the results from multiple attention heads
    def __init__(self):
        super().__init__()
        self.heads=nn.ModuleList([Head() for _ in range(numHeads)]) #Get multiple heads
        self.proj=nn.Linear(featuresLength, featuresLength)
        self.proj.stdDiffer=True
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1)
        out=self.proj(out)
        out=self.dropout(out)
        return out

class Head(nn.Module): #Head of self attention
    def __init__(self):
        super().__init__()
        self.key=nn.Linear(featuresLength, headSize, bias=False) #What I have to share
        self.query=nn.Linear(featuresLength, headSize, bias=False) #What I want to know
        self.value=nn.Linear(featuresLength, headSize, bias=False) #What I will share
        self.register_buffer('tril', torch.tril(torch.ones(contextLength, contextLength)))
        self.dropout=nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        #How much do I want to know about each position?
        # w=q@k.transpose(-2, -1)*C**-0.5
        # w=w.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        # w=F.softmax(w, dim=-1)
        # w=self.dropout(w)
        # out=w@v
        #Flash attention
        out=F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        return out
    
class Block(nn.Module): #A transformer block
    def __init__(self):
        super().__init__()
        self.sa=MultiHead()
        self.ffwd=FeedForward()
        self.ln1=nn.LayerNorm(featuresLength)
        self.ln2=nn.LayerNorm(featuresLength)
    def forward(self, x):
        x=x+self.sa(self.ln1(x)) #Layer norm before computations
        x=x+self.ffwd(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable=nn.Embedding(vocabSize, featuresLength) #Each character has features
        self.positionEmbeddingTable=nn.Embedding(contextLength, featuresLength) #Each position has features.
        self.blocks=nn.Sequential(*[Block() for _ in range(numLayers)])
        self.ln=nn.LayerNorm(featuresLength) #Final layer norm
        self.lm_head=nn.Linear(featuresLength, vocabSize) #Turns the embeddings (features) into logits of vocab size

        #Weight sharing
        self.tokenEmbeddingTable.weight=self.lm_head.weight

        #Init params
        self.apply(self.initWeights)

    def initWeights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'stdDiffer'): #Different init for stuff that has residual layer
                std*=(2*numLayers)**-0.5 #2 times cuz attention and MLP both add to residual layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T=idx.shape

        tokenEmbed=self.tokenEmbeddingTable(idx) #Get the embeddings from the input tokens
        positionEmbed=self.positionEmbeddingTable(torch.arange(T, device=device)) #T*C, features for each position
        x=tokenEmbed+positionEmbed #B, T, C
        x=self.blocks(x)
        x=self.ln(x)
        logits=self.lm_head(x) #Convert them to logits

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
            topProbs, topIndicies=torch.topk(probs, 50, dim=-1)
            #Get prediction
            next=torch.multinomial(topProbs, num_samples=1)
            #Append to the end
            idx=torch.cat((idx, next), dim=1)
        return idx

#Use TF32
torch.set_float32_matmul_precision('high')
model=Model()
model=model.to(device)
optimizer=torch.optim.AdamW(model.parameters(), lr=learningRate, betas=(0.9, 0.95), eps=1e-8)

for iter in range(maxIter):
    t0=time.time()
    #Testing, currently removed to see time
    # if iter%evalInterval==0:
    #     losses=getCurrentLoss()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #Get a batch (not doing the non-repeated until the next block)
    xb, yb=getBatch('train')

    #Training
    with torch.autocast(device_type=device, dtype=torch.bfloat16): #Use bfloat 16 to go even faster
        logits, loss=model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Grad clipping to prevent model shock.
    #Use learning rate scheduler
    lr=getLr(iter)
    for param in optimizer.param_groups:
        param['lr']=lr
    optimizer.step()
    torch.cuda.synchronize() #Wait for GPU to be done
    # if iter%evalInterval==0: #Should be periodic when it runs
    t1=time.time()
    dt=(t1-t0)*1000
    tokenPerSec=batchSize*contextLength/(t1-t0)
    print(f"step: {iter}, loss: {loss:.2f}, dt: {dt:.2f}ms, tok/s: {tokenPerSec:.2f}")

model.eval()
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(tiktoken.decode(model.generate(context, 100)[0].tolist()))