import torch
if torch.cuda.is_available():
    print("ok")
else:
    print("nope")