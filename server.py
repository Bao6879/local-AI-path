from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inferenceStream import loadModel
from inferenceStream import getInitialContext

import time

app = FastAPI() 
model=loadModel()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int=100
    temperature: float=0.8
    topk: int=50

@app.get("/")
def root():
    return {"message": "hello"} 

@app.post("/generate")
async def generate(req: GenerateRequest):
    ctx=getInitialContext(req.prompt)
    gen=model.generate(ctx, tokenCount=req.max_tokens, topk=req.topk, temperature=req.temperature)
    return StreamingResponse(gen, media_type="text/plain")