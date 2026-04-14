import torch
if torch.cuda.is_available():
    print("YES GPU")
else:
    print("NO GPU")


