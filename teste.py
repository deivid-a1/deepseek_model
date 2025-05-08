import torch
print(torch.cuda.get_device_name(0))
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
