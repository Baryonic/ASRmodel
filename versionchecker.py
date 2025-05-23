print("Checking versions of torch, transformers, and torchaudio...")
import torch #type: ignore[import]
import transformers
import torchaudio #type: ignore[import]

print("\033[33mtorch:\033[0m", torch.__version__)
print("\033[33mtransformers:\033[0m", transformers.__version__)
print("\033[33mtorchaudio:\033[0m", torchaudio.__version__)