import torch
import torch.nn as nn


rms_norm = nn.RMSNorm(3)
input = torch.randn(1, 1, 3)
print(input)
output = rms_norm(input)
print(output)