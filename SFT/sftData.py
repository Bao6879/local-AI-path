"""
SFT data pipeline (Step 1).  CPU-only. Run once, validate by eye, then the GPU
session just loads the .pt this produces.

Per example we build, in CAUSAL-LM aligned form (forward() does NOT shift
internally, so the caller must):

    full   = encode(prompt_template) + encode(response) + [EOT_ID]
    idx    = full[:-1]      # model input
    target = full[1:]       # what each position must predict
    labels = target, but every position whose TARGET is a prompt/template token
             is set to -100 so only response tokens (+ EOT) contribute to loss.

The mask boundary is the only subtle part. In target space, target[k] = full[k+1].
A response token sits at full index >= len(prompt). So target[k] is a response
token exactly when k+1 >= len(prompt), i.e. k >= len(prompt) - 1. That boundary
position k = len(prompt)-1 is where the model has just read the final template
token ("### Response:\n") and must emit the FIRST response token. It MUST be
scored. Off-by-one here is the classic silent SFT bug, so the validator below
walks this exact seam.

Padding for batching is NOT done here (that's the Step-2 collate). Because labels
default to -100, padded positions are masked for free when you pad labels with -100.
"""
import json
import argparse
import tiktoken

from sftCommon import buildChatPrompt, EOT_ID

enc = tiktoken.get_encoding('gpt2')
CONTEXT_LENGTH = 1024  # must match inferenceStream.contextLength


def formatExample(instruction: str, output: str):
    """Returns (idx, labels, promptLen) for one instruction/response pair."""
    promptText = buildChatPrompt(instruction)
    promptIds = enc.encode_ordinary(promptText)
    responseIds = enc.encode_ordinary(output.strip()) + [EOT_ID]

    full = promptIds + responseIds
    idx = full[:-1]
    target = full[1:]

    # -100 everywhere by default; only response-token targets get real ids.
    labels = [-100] * len(target)
    firstScored = len(promptIds) - 1          # target index of the 1st response token
    for k in range(firstScored, len(target)):
        labels[k] = target[k]
    return idx, labels, len(promptIds)


def buildDataset(rawPath: str, skipWithInput: bool = True):
    rows = json.load(open(rawPath))
    examples = []
    skippedInput = skippedLong = 0
    for r in rows:
        if skipWithInput and r.get('input', '').strip():
            skippedInput += 1
            continue
        idx, labels, _ = formatExample(r['instruction'], r['output'])
        if len(idx) + 1 > CONTEXT_LENGTH:      # +1 because full == len(idx)+1
            # truncating would chop the EOT and teach it to never stop -> skip whole.
            skippedLong += 1
            continue
        examples.append((idx, labels))
    return examples, skippedInput, skippedLong


def validate(rawPath: str, n: int = 3):
    """Print n formatted examples and WALK THE MASK BOUNDARY by eye."""
    rows = json.load(open(rawPath))
    shown = 0
    for r in rows:
        if r.get('input', '').strip():
            continue
        idx, labels, promptLen = formatExample(r['instruction'], r['output'])
        target = idx[1:] + [labels[-1] if labels[-1] != -100 else EOT_ID]  # for display only
        # reconstruct full to decode cleanly
        full = enc.encode_ordinary(buildChatPrompt(r['instruction'])) \
               + enc.encode_ordinary(r['output'].strip()) + [EOT_ID]

        nScored = sum(1 for x in labels if x != -100)
        nMasked = sum(1 for x in labels if x == -100)
        b = promptLen - 1   # boundary index in idx/labels space

        print("=" * 72)
        print(f"EXAMPLE {shown+1}")
        print(f"  instruction : {r['instruction']!r}")
        print(f"  output      : {r['output'].strip()[:80]!r}{'...' if len(r['output'].strip())>80 else ''}")
        print(f"  total tokens (idx): {len(idx)}   prompt tokens: {promptLen}")
        print(f"  scored (response+EOT): {nScored}   masked (prompt/template): {nMasked}")
        print("  --- boundary walk (the off-by-one zone) ---")
        # one token BEFORE boundary: last masked prompt position
        print(f"    labels[{b-1}] = {labels[b-1]:>6}  (input tok {idx[b-1]!r}->{enc.decode([idx[b-1]])!r})   <- still MASKED, must be -100")
        # boundary: last template token in, first response token as target
        print(f"    labels[{b}] = {labels[b]:>6}  (input tok {enc.decode([idx[b]])!r} -> target {enc.decode([labels[b]])!r})   <- FIRST SCORED, = 1st response token")
        print(f"    labels[{b+1}] = {labels[b+1]:>6}  (target {enc.decode([labels[b+1]])!r})   <- scored")
        # tail: last scored target must be EOT
        print(f"    labels[-1] = {labels[-1]:>6}  -> {'<|endoftext|> ✓ STOP TOKEN' if labels[-1]==EOT_ID else 'NOT EOT — BUG'}")
        # assertions that must hold
        assert labels[b-1] == -100, "boundary-1 should be masked"
        assert labels[b] == full[promptLen], "first scored should be first response token"
        assert labels[b] != -100, "first response token must be scored"
        assert labels[-1] == EOT_ID, "sequence must end scored on EOT"
        shown += 1
        if shown >= n:
            break
    print("=" * 72)
    print("All boundary assertions passed.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='alpaca_data.json')
    ap.add_argument('--out', default='sft_tokenized.pt')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()

    validate(args.data, n=3)

    if not args.validate_only:
        examples, skippedInput, skippedLong = buildDataset(args.data)
        print(f"\nbuilt {len(examples)} examples "
              f"(dropped {skippedInput} with input, {skippedLong} over {CONTEXT_LENGTH} tok)")
        import pickle
        from sftCommon import PROMPT_TEMPLATE
        with open(args.out, 'wb') as f:
            pickle.dump({'examples': examples,         # list of (idx:list[int], labels:list[int])
                         'template': PROMPT_TEMPLATE,
                         'eot': EOT_ID}, f)
        print(f"saved -> {args.out}  (load with pickle; collate -> tensors in Step 2)")
