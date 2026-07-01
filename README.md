---
title: Small Finest Web LLM
colorFrom: gray
colorTo: blue

sdk: docker
app_port: 7860
pinned: false


---

# small-finest-web

A ~200M-parameter language model trained from scratch on FineWeb, served as a
text completer. Enter a prompt and the completion streams back token by token.

The model weights are pulled from the [Bao6879/small-finest-web](https://huggingface.co/Bao6879/small-finest-web)
repo at startup. 