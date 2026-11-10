import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# macbook : "mps"
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print(f"Running on Apple GPU (MPS): {device}")
else:
    print("Running on CPU")

# Hyperparameters
DROPOUT = 0.2
BATCH_SIZE = 16
CONTEXT_LENGTH = 256
DIM = 256
N_HEADS = 8
N_LAYERS = 6
MAX_ITERS = 6000
EVAL_INTERVAL = 250
LR = 3e-4

