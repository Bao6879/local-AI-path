from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import codecs

from inferenceStream import loadModel, getInitialContext, enc

app = FastAPI() 
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