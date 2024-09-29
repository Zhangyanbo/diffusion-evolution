import torch
import torch.nn as nn


class RandomProjection(nn.Module):
    def __init__(self, in_features, out_features, normalize=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.normalize = normalize
        self.init_weight()
    
    def init_weight(self):
        self.linear.weight.data = torch.randn_like(self.linear.weight.data) / (self.in_features ** 0.5)
        if self.normalize:
            self.linear.weight.data /= self.linear.weight.data.norm(dim=1, keepdim=True)

    def forward(self, x):
        return self.linear(x)