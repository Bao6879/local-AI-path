from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from fastapi.concurrency import run_in_threadpool
import json
import os
import codecs
import torch

torch.set_num_threads(4)

from inferenceStream import loadModel, getInitialContext, enc
from sftCommon import buildChatPrompt, EOT_ID   # shared with training -> no template drift

app = FastAPI()

REPO_ID = "Bao6879/small-finest-web"
CKPT_DIR = "checkpoints"

# ---------------------------------------------------------------------------
# REGISTRY — one entry per selectable snapshot. This is the ONLY place to edit
# when you add/rename checkpoints. `kind` drives behavior (completer = raw
# prompt, runs to max_tokens; chat = Alpaca-wrapped, stops on EOT). `file` is
# the filename in the HF repo. `label` is what the UI shows.
# Order here = order the UI dropdown will list them.
# ---------------------------------------------------------------------------
REGISTRY = {
    # --- completer snapshots (base / pretraining): no template, no stop ---
    "completer-v0":  {"file": "trainingCheckpoints/ckpt_pre_step0.pt", "kind": "completer", "label": "Completer — step 0"},
    "completer-v1":  {"file": "trainingCheckpoints/ckpt_pre_step750.pt", "kind": "completer", "label": "Completer — step 750"},
    "completer-v2":  {"file": "trainingCheckpoints/ckpt_pre_step3000.pt", "kind": "completer", "label": "Completer — step 3000"},
    "completer-v3":  {"file": "trainingCheckpoints/ckpt_pre_step13000.pt", "kind": "completer", "label": "Completer — step 13000"},
    "completer-final": {"file": "trainingCheckpoints/ckpt.pt", "kind": "completer", "label": "Completer — final, 19073 steps"},
    # ... up to ~5; fill in your real pretraining-snapshot filenames

    # --- chat snapshots (SFT): Alpaca template + stop on EOT ---
    "chat-v0": {"file": "sftCheckpoints/ckpt_sft_step0.pt", "kind": "chat",      "label": "Chat — step 0"},
    "chat-v1": {"file": "sftCheckpoints/ckpt_sft_step25.pt", "kind": "chat",      "label": "Chat — step 25"},
    "chat-v2": {"file": "sftCheckpoints/ckpt_sft_step100.pt", "kind": "chat",      "label": "Chat — step 100"},
    "chat-v3": {"file": "sftCheckpoints/ckpt_sft_step700.pt", "kind": "chat",      "label": "Chat — step 700"},
    "chat-final": {"file": "sftCheckpoints/ckpt_sft_final.pt",   "kind": "chat",      "label": "Chat — final, 2937 steps"},
    # ... up to ~5; these come out of the Step-2 training loop's checkpoints
}
DEFAULT_MODEL = "chat-final"   # default => current behavior; old clients keep working

# ---------------------------------------------------------------------------
# TEST TOGGLE. Before the real SFT / intermediate weights exist, load EVERY
# entry from the base ckpt.pt. This exercises routing + template wrapping +
# stop logic with only one checkpoint present. The "chat" entries won't behave
# like a chatbot (they're the base model), but every code path runs. When the
# real files are uploaded to the repo, flip this to False — no code change.
# ---------------------------------------------------------------------------
SERVE_FALLBACK_ONLY = False
FALLBACK_FILE = "ckpt.pt"

# ---------------------------------------------------------------------------
# Lazy loader with a resident cap. Models load on first use and stay cached —
# this is NOT per-request reloading (the naive mistake), it's load-once. With
# ~800MB fp32 each, 10 models ~= 8GB into 16GB fits, so MAX_RESIDENT=None
# (keep all) is fine. Set an int if RAM ever feels tight; least-recently-added
# is evicted.
# ---------------------------------------------------------------------------
MAX_RESIDENT = None
_loaded = {}   # key -> Model (insertion-ordered)

class LoadRequest(BaseModel):
    model: str

@app.post("/load")                       # the visible load beat
async def load(req: LoadRequest):
    if req.model not in REGISTRY:
        raise HTTPException(400, f"unknown model {req.model!r}")
    await run_in_threadpool(getModel, req.model)   # ~5s, off the event loop
    return {"status": "resident", "model": req.model}

STATS_FILES = {"completer": "pretrain_stats.json", "chat": "sft_stats.json"}

def _downsample(curve, n=300):
    if len(curve) <= n: return curve
    s = len(curve) / n
    return [curve[int(i*s)] for i in range(n)]

@app.get("/metadata")                    # feeds the Task Manager panel
async def metadata():
    out = {}
    for kind, path in STATS_FILES.items():
        if os.path.exists(path):
            with open(path) as f: data = json.load(f)
            data["curve"] = _downsample(data["curve"])
            out[kind] = data
    return out

def _ensure_downloaded(filename):
    path = os.path.join(CKPT_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(CKPT_DIR, exist_ok=True)
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=CKPT_DIR)
    return path


def getModel(key):
    if key in _loaded:
        return _loaded[key]
    filename = FALLBACK_FILE if SERVE_FALLBACK_ONLY else REGISTRY[key]["file"]
    path = _ensure_downloaded(filename)
    model = loadModel(path)
    if MAX_RESIDENT is not None and len(_loaded) >= MAX_RESIDENT:
        _loaded.pop(next(iter(_loaded)))   # evict oldest
    _loaded[key] = model
    return model


# --- boot: download everything up front (slow boot, snappy session), then
# load only the default into RAM. To defer downloads to first use instead,
# comment out the predownload loop.
for _key, _entry in REGISTRY.items():
    try:
        _ensure_downloaded(FALLBACK_FILE if SERVE_FALLBACK_ONLY else _entry["file"])
    except Exception as e:
        print(f"warn: could not pre-download {_key}: {e}")
getModel(DEFAULT_MODEL)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    topk: int = 50
    model: str = DEFAULT_MODEL          # "completer-final" | "chat-final" | ...


@app.get("/")
async def ui():
    return FileResponse("ui.html")

@app.get("/models")
async def list_models():
    return [{"id": k, "kind": v["kind"], "label": v["label"], "file": v["file"]} for k, v in REGISTRY.items()]


async def token_stream(req):
    entry = REGISTRY[req.model]
    kind = entry["kind"]
    model = getModel(req.model)

    # the seam: chat wraps the prompt with the SAME function training used.
    prompt = buildChatPrompt(req.prompt) if kind == "chat" else req.prompt
    ctx = getInitialContext(prompt)

    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    for token in model.generate(ctx, tokenCount=req.max_tokens, topk=req.topk, temperature=req.temperature):
        if token == EOT_ID:
            if kind == "chat":
                break          # stop the answer cleanly (before decode/yield)
            continue           # completer: skip the stop token, keep going to max_tokens
        text = decoder.decode(enc.decode_single_token_bytes(token))
        if text:
            yield text
    tail = decoder.decode(b'', final=True)
    if tail:
        yield tail


@app.post("/generate")
async def generate(req: GenerateRequest):
    if req.model not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
    return StreamingResponse(token_stream(req), media_type="text/plain")