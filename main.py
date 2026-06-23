from fastapi import FastAPI
import time

app = FastAPI() 
start=time.time()

@app.get("/")
def root():
    return {"message": "hello"} 

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"model": "mini-llm", "params": "200M"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/uptime")
def uptime():
    timePassed=time.time()-start
    return {"uptime": round(timePassed, 2)}