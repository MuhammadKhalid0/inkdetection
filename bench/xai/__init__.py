"""Inkalyzer-based input-space information-bottleneck (IBA) XAI for the ink models.

Vendored + adapted from https://github.com/younader/Inkalyzer (Y. Nader).
Adaptations for this project:
  * depth dimension parameterised (upstream hardcodes 30; our models use 32).
  * autocast dtype configurable (bf16 by default; upstream defaults to fp16
    which overflows on our t192 warp chains).
  * notebook-only bits (IPython/matplotlib animation, tqdm.notebook) removed.
"""
