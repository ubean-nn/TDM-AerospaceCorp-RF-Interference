#!/usr/bin/env python3

import torch
if torch.cuda.is_available():
 print("Have GPU!")
else:
 print("No GPU")
