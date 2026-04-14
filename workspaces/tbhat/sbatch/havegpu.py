

import torch
if torch.cuda.is_available():
    print("Have GPU")
else:
    print("No GPU")