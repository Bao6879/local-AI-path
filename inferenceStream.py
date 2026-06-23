import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

vocabSize=50304
numLayers=23
dropout=0.0
contextLength=1024
featuresLength=768
numHeads=12
headSize=64
realVocabSize=50257  # tiktoken gpt2 vocab; ids above this are padding
ckptPath='checkpoints/ckpt.pt'

device='cuda' if torch.cuda.is_available() else 'cpu'
enc=tiktoken.get_encoding('gpt2')

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        projLayer=nn.Linear(featuresLength*4, featuresLength)
        self.net=nn.Sequential(
            nn.Linear(featuresLength, featuresLength*4),
            nn.GELU(approximate='tanh'),
            projLayer,
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(featuresLength, 3*featuresLength, bias=False)
        self.proj=nn.Linear(featuresLength, featuresLength)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x):
        B, T, C=x.shape
        q, k, v=self.qkv(x).split(featuresLength, dim=2)
        q=q.view(B, T, numHeads, headSize).transpose(1, 2)
        k=k.view(B, T, numHeads, headSize).transpose(1, 2)
        v=v.view(B, T, numHeads, headSize).transpose(1, 2)
        out=F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out=out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa=MultiHead()
        self.ffwd=FeedForward()
        self.ln1=nn.LayerNorm(featuresLength)
        self.ln2=nn.LayerNorm(featuresLength)
    def forward(self, x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable=nn.Embedding(vocabSize, featuresLength)
        self.positionEmbeddingTable=nn.Embedding(contextLength, featuresLength)
        self.blocks=nn.Sequential(*[Block() for _ in range(numLayers)])
        self.ln=nn.LayerNorm(featuresLength)
        self.lm_head=nn.Linear(featuresLength, vocabSize)
        self.tokenEmbeddingTable.weight=self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T=idx.shape
        tokenEmbed=self.tokenEmbeddingTable(idx)
        positionEmbed=self.positionEmbeddingTable(torch.arange(T, device=idx.device))
        x=tokenEmbed+positionEmbed
        x=self.blocks(x)
        x=self.ln(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, tokenCount, topk=50, temperature=1.0):
        for _ in range(tokenCount):
            idxCropped=idx[:, -contextLength:]
            logits, _=self(idxCropped)
            logits=logits[:, -1, :realVocabSize]      # last step, real vocab only
            logits=logits/max(temperature, 1e-6)       # <1 = sharper/safer, >1 = wilder
            probs=F.softmax(logits, dim=-1)
            topProbs, topIdx=torch.topk(probs, topk, dim=-1)
            choice=torch.multinomial(topProbs, num_samples=1)
            nextTok=torch.gather(topIdx, -1, choice)
            yield enc.decode(nextTok[0].tolist())
            idx=torch.cat((idx, nextTok), dim=1)


def loadModel(path=ckptPath):
    ckpt=torch.load(path, map_location=device, weights_only=False)
    state=ckpt['model']
    state={k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model=Model().to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {path} | trained to iter {ckpt.get('iter','?')} | last loss {ckpt.get('loss','?')}")
    return model


def getInitialContext(prompt=""):
    ids=enc.encode_ordinary(prompt)
    return torch.tensor([ids], device=device)

if __name__=='__main__':
    model=loadModel()

    MAX_NEW_TOKENS=120
    TEMPERATURE=0.8
    TOPK=50

    print("\nType a prompt and press Enter. The model continues it.")

    while True:
        try:
            prompt=input("input> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not prompt:
            continue
        if prompt in ('/quit', '/exit', '/q'):
            print("bye")
            break

        ctx=getInitialContext(prompt)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            gen=model.generate(ctx, tokenCount=MAX_NEW_TOKENS, topk=TOPK, temperature=TEMPERATURE)
            for token in gen:
                print(token, end="", flush=True)

