"""
Step 2 — SFT training loop.

Loads the base completer checkpoint and continues training it on the
Alpaca-formatted, loss-masked data from sftData.py, producing a sequence of
"chat" snapshots that show the model learning to ANSWER rather than CONTINUE.

Design decisions carried from planning:
  * Loss masking is FREE: forward() calls F.cross_entropy whose default
    ignore_index is -100, and the data already carries -100 on every
    prompt/template token. The collate just has to extend that -100 to padding.
  * Right-padding + causal attention needs NO attention mask: a real token at
    position t only attends to <=t, so it never sees padding (which sits to its
    right), and padded positions are masked out of the loss. So we pad, set pad
    targets to -100, and that's it.
  * Checkpoints are saved STRIPPED ({'model', 'iter', 'loss'} — no optimizer),
    so loadModel reads them directly and they're ~800MB not ~2.4GB.
  * Checkpoint cadence is FRONT-LOADED: format-following emerges early then
    plateaus, so dense-early snapshots catch the transition; uniform spacing
    would waste snapshots on a flat tail.
  * At every checkpoint we also run a fixed PROBE SET and append the outputs to
    a transcript. That transcript is the evolution demo and survives even if the
    heavy intermediate weights are later deleted.

LR guidance: SFT uses a MUCH lower LR than pretraining (1e-5..5e-5, err low —
too high wrecks the base knowledge). Default 2e-5 with short warmup + cosine.

Run order: smoke test on CPU FIRST (`--smoke`), then the real run on GPU.
"""
import os
import math
import time
import json
import pickle
import random
import argparse

import torch
from torch.nn.utils import clip_grad_norm_

from inferenceStream import loadModel, getInitialContext, enc, device
from sftCommon import buildChatPrompt, EOT_ID

# ---- defaults (all overridable via argparse) -------------------------------
DEFAULTS = dict(
    base="checkpoints/ckpt.pt",
    data="sft_tokenized.pkl",
    out="checkpoints_sft",
    epochs=3,
    batch_size=32,
    lr=2e-5,
    min_lr_frac=0.1,        # cosine floor = lr * this
    warmup_steps=100,
    weight_decay=0.01,
    grad_clip=1.0,
    ema_beta=0.95,          # loss smoothing for the curve + checkpoint dots
    seed=1337,
)

# Fixed probe prompts. The SAME prompts at every checkpoint => a clean
# step-by-step transcript of the model learning to answer. Swap these for
# whatever best tells your story; keep them fixed across the run.
PROBE_PROMPTS = [
    "Hello!",
    "What is the capital of France?",
    "List three tips for staying healthy.",
    "Write a one-sentence summary of what a computer is.",
    "Explain why the sky is blue.",
]

# Front-loaded checkpoint schedule (in optimizer steps). Dense early, sparse
# later. End-of-epoch and final are always saved on top of these. Keep them all
# during the run; curate down to ~5 for the demo afterwards.
CHECKPOINT_STEPS = [0, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000]


class StatsLogger:
    """Emits the JSON the phase-4 HTML reads. The whole point is that a
    checkpoint's dot lands ON the loss curve: both the curve point and the
    checkpoint vital read the SAME smoothed loss (self.ema) at the SAME x, so
    they can't disagree. Raw per-batch loss is spiky; we never store it.

    Schema (uniform with the eventual pretraining stats, branch on x_axis):
      {
        "kind": "sft", "x_axis": "step",
        "total_steps": <denominator for pct_complete>,
        "total_tokens": null,                 # set for pretraining instead
        "curve": [[x, smoothed_loss], ...],
        "checkpoints": [
          {"step","tokens_seen","loss","lr","tag","file"}, ...
        ]
      }
    """
    def __init__(self, path, kind, x_axis, total_steps=None, total_tokens=None, ema_beta=0.95):
        self.path = path
        self.beta = ema_beta
        self.ema = None
        self.curve = []
        self.checkpoints = []
        self.meta = {"kind": kind, "x_axis": x_axis,
                     "total_steps": total_steps, "total_tokens": total_tokens}

    def update(self, raw_loss):
        self.ema = raw_loss if self.ema is None else self.beta * self.ema + (1 - self.beta) * raw_loss
        return self.ema

    def log_point(self, x):
        self.curve.append([int(x), round(self.ema, 5)])

    def log_checkpoint(self, step, tokens_seen, lr, tag, file):
        self.checkpoints.append({
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "loss": round(self.ema, 5),     # SAME smoothed value as the curve -> dot on the line
            "lr": float(lr),
            "tag": tag,
            "file": file,
        })

    def flush(self):
        with open(self.path, "w") as f:
            json.dump({**self.meta, "curve": self.curve, "checkpoints": self.checkpoints}, f, indent=2)


def collate(batch, pad_id=EOT_ID):
    """batch: list of (idx:list[int], labels:list[int]) -> padded (input, target) tensors.
    Pads inputs with pad_id (cosmetic — masked) and targets with -100 (ignored)."""
    maxlen = max(len(idx) for idx, _ in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    targets = torch.full((B, maxlen), -100, dtype=torch.long)
    for i, (idx, lab) in enumerate(batch):
        L = len(idx)
        input_ids[i, :L] = torch.tensor(idx, dtype=torch.long)
        targets[i, :L] = torch.tensor(lab, dtype=torch.long)
    return input_ids, targets


def get_lr(step, total_steps, base_lr, warmup, min_lr_frac):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total_steps:
        return base_lr * min_lr_frac
    prog = (step - warmup) / max(1, total_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


@torch.no_grad()
def probe(model, prompts=PROBE_PROMPTS, max_tokens=60, temperature=0.8, topk=50):
    """Run the fixed prompts through the CURRENT model via the chat path
    (template-wrapped, stop on EOT) and return (prompt, completion) pairs."""
    was_training = model.training
    model.eval()
    use_autocast = (device == 'cuda')
    outs = []
    for p in prompts:
        ctx = getInitialContext(buildChatPrompt(p))
        ids = []
        ctxmgr = torch.autocast(device_type=device, dtype=torch.bfloat16) if use_autocast \
            else torch.no_grad()
        with ctxmgr:
            for tok in model.generate(ctx, tokenCount=max_tokens, topk=topk, temperature=temperature):
                if tok == EOT_ID:
                    break
                ids.append(tok)
        outs.append((p, enc.decode(ids)))
    if was_training:
        model.train()
    return outs


def save_checkpoint(model, step, loss, out_dir, tag):
    """STRIPPED save: model weights + scalars only, no optimizer state.
    loadModel handles the '_orig_mod.' prefix if the model was torch.compile'd."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_sft_{tag}.pt")
    torch.save({'model': model.state_dict(),
                'iter': int(step),
                'loss': float(loss)}, path)
    return path


def log_probe(transcript_path, step, results):
    with open(transcript_path, "a") as f:
        f.write(f"\n{'='*70}\nSTEP {step}\n{'='*70}\n")
        for p, out in results:
            f.write(f"  >>> {p}\n      {out!r}\n")


def train(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with open(cfg.data, "rb") as f:
        data = pickle.load(f)["examples"]
    print(f"loaded {len(data)} examples from {cfg.data}")

    model = loadModel(cfg.base)
    model.train()
    if getattr(cfg, "compile", False):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(data) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    print(f"{steps_per_epoch} steps/epoch x {cfg.epochs} epochs = {total_steps} steps")

    os.makedirs(cfg.out, exist_ok=True)
    transcript = os.path.join(cfg.out, "probe_transcript.txt")
    open(transcript, "w").close()
    stats = StatsLogger(os.path.join(cfg.out, "sft_stats.json"),
                        kind="sft", x_axis="step",
                        total_steps=total_steps, total_tokens=None,
                        ema_beta=cfg.ema_beta)
    use_autocast = (device == 'cuda')

    step = 0
    tokens_seen = 0
    lr = cfg.lr
    ckpt_set = set(CHECKPOINT_STEPS)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        random.shuffle(data)
        for bi in range(0, len(data), cfg.batch_size):
            batch = data[bi:bi + cfg.batch_size]
            input_ids, targets = collate(batch)
            input_ids, targets = input_ids.to(device), targets.to(device)

            lr = get_lr(step, total_steps, cfg.lr, cfg.warmup_steps, cfg.min_lr_frac)
            for g in opt.param_groups:
                g['lr'] = lr

            # forward FIRST: gives this model's loss, incl. the base model at step 0
            if use_autocast:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(input_ids, targets)
            else:
                _, loss = model(input_ids, targets)
            raw = loss.item()
            smoothed = stats.update(raw)
            stats.log_point(step)

            # checkpoint BEFORE the optimizer step => saved model = "after `step` updates";
            # tokens_seen (bumped after the step) matches; stored loss is the SAME smoothed
            # number as the curve point at this x, so the dot lands on the line.
            if step in ckpt_set:
                p = save_checkpoint(model, step, smoothed, cfg.out, f"step{step}")
                log_probe(transcript, step, probe(model))
                stats.log_checkpoint(step, tokens_seen, lr, f"step{step}", os.path.basename(p))
                stats.flush()
                print(f"[ckpt] step {step} -> {p}  (smoothed {smoothed:.4f})")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            tokens_seen += sum(len(idx) for idx, _ in batch)   # real tokens, excludes padding
            if step % 20 == 0:
                dt = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps} raw {raw:.4f} "
                      f"ema {smoothed:.4f} lr {lr:.2e} ({dt:.0f}s)")
            step += 1

        # always snapshot at end of each epoch
        p = save_checkpoint(model, step, stats.ema, cfg.out, f"epoch{epoch+1}")
        log_probe(transcript, step, probe(model))
        stats.log_checkpoint(step, tokens_seen, lr, f"epoch{epoch+1}", os.path.basename(p))
        stats.flush()
        print(f"[ckpt] end epoch {epoch+1} -> {p}")

    p = save_checkpoint(model, step, stats.ema, cfg.out, "final")
    log_probe(transcript, step, probe(model))
    stats.log_checkpoint(step, tokens_seen, lr, "final", os.path.basename(p))
    stats.flush()
    print(f"[done] final -> {p}\nstats -> {stats.path}\ntranscript -> {transcript}")


def smoke(cfg):
    """CPU sanity check: overfit 4 examples. Loss should crater toward ~0 in
    ~100 steps. Then save+reload to prove the checkpoint round-trips. If loss
    moves and reload matches, the loop + masking + saving are wired right."""
    torch.manual_seed(0)
    with open(cfg.data, "rb") as f:
        data = pickle.load(f)["examples"][:4]
    print(f"SMOKE: overfitting {len(data)} examples on {device}")

    model = loadModel(cfg.base)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # higher LR: we WANT overfit

    input_ids, targets = collate(data)
    input_ids, targets = input_ids.to(device), targets.to(device)

    first = last = None
    for s in range(100):
        _, loss = model(input_ids, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s == 0:
            first = loss.item()
        if s % 10 == 0:
            print(f"  step {s:3d}  loss {loss.item():.4f}")
        last = loss.item()
    print(f"SMOKE loss: {first:.4f} -> {last:.4f}  "
          f"({'OK: loss collapsed' if last < first * 0.3 else 'SUSPICIOUS: barely moved'})")

    path = save_checkpoint(model, 100, last, cfg.out, "smoke")
    reloaded = loadModel(path)
    with torch.no_grad():
        _, l2 = reloaded(input_ids, targets)
    print(f"reload check: saved loss {last:.4f} vs reloaded {l2.item():.4f}  "
          f"({'OK' if abs(l2.item() - last) < 1e-3 else 'MISMATCH'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    (smoke if cfg.smoke else train)(cfg)
