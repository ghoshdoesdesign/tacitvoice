import os

# Must be set before torch is imported: beam search uses a few ops that Apple's
# GPU backend lacks, and this lets them fall back to the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
