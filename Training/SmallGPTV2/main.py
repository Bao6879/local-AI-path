import torch
import time
import math
import os
import glob
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

#Setup
enc=tiktoken.get_encoding('gpt2')
dataDir='data'
shards=sorted(glob.glob(os.path.join(dataDir, 'edu_fineweb_train_*.bin')))

trainShards=shards[:-1]
testShards=shards[-1:]

#Settings
vocabSize=50304
numLayers=23
dropout=0.0
totalBatchSize=524288 #Full batch size (in tokens)
batchSize=16 #At once, micro batch size
contextLength=1024 #Up to this many tokens for predictions
gradAccumSteps=totalBatchSize//(batchSize*contextLength) #Number of steps
featuresLength=768 #Features for each token
numHeads=12 #Num heads*head size = feature length.
headSize=64
maxIter=19073 #~1 full epoch over the 10B-token sample
evalInterval=100 #Every interval, run evaluation
checkpointInterval=1000 #Save a single rolling checkpoint this often (overwrites the previous one)
learningRate=6e-4
seed=3108

device='cuda' if torch.cuda.is_available() else 'cpu' #Use GPU to accelerate process if possible
torch.manual_seed(seed=seed)

#Cosine learning rate decay with warmup (like GPT-3)
minLr=learningRate*0.1
warmUp=maxIter*0.05
def getLr(i):
    if i<warmUp:
        return learningRate*(i+1)/warmUp
    decayRate=(i-warmUp)/(maxIter-warmUp)
    coeff=0.5*(1.0+math.cos(math.pi*decayRate))
    return minLr+coeff*(learningRate-minLr)

def getBatch(split):
    shardList=trainShards if split=='train' else testShards
    shard=np.random.choice(shardList)
    data=np.memmap(shard, dtype=np.uint16, mode='r')
    ix=torch.randint(len(data)-contextLength, (batchSize,))
    x=torch.stack([torch.from_numpy(data[i:i+contextLength].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(data[i+1:i+contextLength+1].astype(np.int64)) for i in ix])
    x, y=x.to(device), y.to(device)
    return x, y

def getCurrentLoss():
    model.eval()
    with torch.no_grad():
        testLossAccum=0.0
        testSteps=20
        for _ in range (testSteps):
            x, y=getBatch('test')
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss=model(x, y)
            loss=loss/testSteps
            testLossAccum+=loss.detach()
        print(f"Test loss: {testLossAccum:.4f}")

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

class MultiHead(nn.Module): #Batched multi-head self-attention (single QKV matmul, one flash-attention call)
    def __init__(self):
        super().__init__()
        #One combined projection produces Q, K and V for every head at once (768 -> 3*768)
        self.qkv=nn.Linear(featuresLength, 3*featuresLength, bias=False)
        self.proj=nn.Linear(featuresLength, featuresLength)
        self.proj.stdDiffer=True #Residual-output layer: gets the scaled init
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        B, T, C=x.shape
        q, k, v=self.qkv(x).split(featuresLength, dim=2) #Each is (B, T, C)
        #Reshape (B, T, C) -> (B, numHeads, T, headSize) so attention runs over all heads in parallel
        q=q.view(B, T, numHeads, headSize).transpose(1, 2)
        k=k.view(B, T, numHeads, headSize).transpose(1, 2)
        v=v.view(B, T, numHeads, headSize).transpose(1, 2)
        #One flash-attention kernel for the whole batch of heads; scales and causal-masks internally
        out=F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        out=out.transpose(1, 2).contiguous().view(B, T, C) #Re-assemble heads back into (B, T, C)
        out=self.dropout(self.proj(out))
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
            nextIndex=torch.multinomial(topProbs, num_samples=1)
            next=torch.gather(topIndicies, -1, nextIndex)
            #Append to the end
            idx=torch.cat((idx, next), dim=1)
        return idx

#Use TF32
torch.set_float32_matmul_precision('high')
model=Model()
model=model.to(device)
optimizer=torch.optim.AdamW(model.parameters(), lr=learningRate, betas=(0.9, 0.95), eps=1e-8)
useCompile=True
if useCompile:
    try:
        model=torch.compile(model)
        warmX=torch.randint(0, vocabSize, (batchSize, contextLength), device=device)
        warmY=torch.randint(0, vocabSize, (batchSize, contextLength), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, warmLoss=model(warmX, warmY)
        warmLoss.backward()
        model.zero_grad(set_to_none=True)
        print("torch.compile: enabled")
    except Exception as e:
        model=getattr(model, '_orig_mod', model) #Unwrap the compiled wrapper, keep training eager
        model.zero_grad(set_to_none=True)
        print(f"torch.compile: disabled, falling back to eager ({type(e).__name__}: {e})")

for iter in range(maxIter):
    if iter%evalInterval==0:
        losses=getCurrentLoss()
    #Training
    model.train() #Eval flips model to eval mode; flip back so dropout stays active
    t0=time.time() #Time the training step to report throughput
    optimizer.zero_grad(set_to_none=True)
    lossAccum=0.0
    for j in range(gradAccumSteps):
        #Get a batch (not doing the non-repeated until the next block)
        xb, yb=getBatch('train')
        with torch.autocast(device_type=device, dtype=torch.bfloat16): #Use bfloat 16 to go even faster
            logits, loss=model(xb, yb)
        #Scale loss to account for grad accum
        loss=loss/gradAccumSteps
        lossAccum+=loss.detach()
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Grad clipping to prevent model shock.
    #Use learning rate scheduler
    lr=getLr(iter)
    for param in optimizer.param_groups:
        param['lr']=lr
    optimizer.step()
    if device=='cuda':
        torch.cuda.synchronize() #GPU work is async; wait for it to finish before timing
    dt=time.time()-t0
    tokensPerSec=totalBatchSize/dt #totalBatchSize tokens processed per optimizer step
    print(f"step {iter:4d} | loss {lossAccum:.4f} | lr {lr:.2e} | norm {norm:.2f} | dt {dt*1000:.0f}ms | tok/s {tokensPerSec:,.0f}")
    if iter>0 and iter%checkpointInterval==0:
        os.makedirs('checkpoints', exist_ok=True)
        ckptPath=os.path.join('checkpoints', 'ckpt.pt')
        tmpPath=ckptPath+'.tmp'
        torch.save({
            'iter': iter,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': lossAccum.item(),
        }, tmpPath)
        os.replace(tmpPath, ckptPath)

#Final checkpoint after training completes, overwrite the same single file
os.makedirs('checkpoints', exist_ok=True)
ckptPath=os.path.join('checkpoints', 'ckpt.pt')
tmpPath=ckptPath+'.tmp'
torch.save({
    'iter': maxIter,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': lossAccum.item(),
}, tmpPath)
os.replace(tmpPath, ckptPath)

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
tokens=model.generate(context, 100)[0].tolist()
decoded=enc.decode(tokens=tokens)
print(decoded)