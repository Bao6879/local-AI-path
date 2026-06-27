"""
Shared SFT constants + prompt builder.

This file is the single source of truth for the chat format. BOTH the training
data pipeline (sftData.py) and the serving layer (server.py chat path) import
from here, so the train-time template and the serve-time template are physically
the same string and cannot drift. A train/serve template mismatch is the #1 way
SFT "mysteriously doesn't work", so we make drift impossible by construction.
"""

# tiktoken gpt2 <|endoftext|>. encode_ordinary() will NOT emit this from text
# (it tokenizes the literal characters), so it is appended as a raw int.
# generate() in inferenceStream.py slices logits[..., :realVocabSize] == [:50257],
# which INCLUDES index 50256, so the model can actually sample it as a stop token.
EOT_ID = 50256

# Alpaca no-input format, with the "Below is an instruction..." preamble
# DELIBERATELY DROPPED: at ~200M it buys nothing and just eats context. This is
# an intentional choice, recorded here so it reads as a decision, not an omission.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def buildChatPrompt(instruction: str) -> str:
    """The exact text placed before the response.

    Training uses it to locate where the prompt ends (the loss-mask boundary).
    Serving uses it to wrap a user's prompt in chat mode before encoding.
    Identical call on both sides => identical tokenization => no drift.
    """
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
