import os

# Ops missing on Apple's MPS backend run on the CPU instead of failing.
# Read when torch loads, so it is set here first.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
