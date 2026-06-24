from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import os
import codecs
import torch

torch.set_num_threads(4)

from inferenceStream import loadModel, getInitialContext, enc

app = FastAPI()

if not os.path.exists("checkpoints/ckpt.pt"):
    os.makedirs("checkpoints", exist_ok=True)
    hf_hub_download(
        repo_id="Bao6879/small-finest-web",
        filename="ckpt.pt",
        local_dir="checkpoints/"
    )
model=loadModel()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int=100
    temperature: float=0.8
    topk: int=50

@app.get("/")
async def ui():
    return FileResponse("ui.html")

async def token_stream(req):
    decoder=codecs.getincrementaldecoder('utf-8')('replace')
    ctx=getInitialContext(req.prompt)
    for token in model.generate(ctx, tokenCount=req.max_tokens, topk=req.topk, temperature=req.temperature):
        token_bytes=enc.decode_single_token_bytes(token)
        text = decoder.decode(token_bytes)
        if text:
            yield text
    tail=decoder.decode(b'', final=True)
    if tail:
        yield tail

@app.post("/generate")
async def generate(req: GenerateRequest):
    return StreamingResponse(token_stream(req), media_type="text/plain")