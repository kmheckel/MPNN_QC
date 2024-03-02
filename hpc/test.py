import torch
assert torch.cuda.is_available(), "WARNING! You are running on a non-GPU instance. For this practical a GPU is highly recommended."
REQUIRED_VERSION = "2.1.0+cu121"
TORCH_VERSION = torch.__version__
CUDA_VERSION = TORCH_VERSION.split("+")

if TORCH_VERSION != REQUIRED_VERSION:
  print(f"Detected torch version {TORCH_VERSION}, but notebook was created for {REQUIRED_VERSION}")
  print(f"Attempting installation of {REQUIRED_VERSION}")
#   !pip install torch==2.1.0+cu121
print("Correct version of torch detected. You are running on a machine with GPU.")