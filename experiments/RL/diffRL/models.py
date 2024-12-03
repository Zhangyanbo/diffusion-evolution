import torch
import torch.nn as nn


class ControllerMLP(nn.Module):
    def __init__(self, dim_in, dim_out, n_hidden, n_hidden_layers=1):
        super().__init__()
        hidden_layers = []
        for _ in range(n_hidden_layers-1):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, n_hidden),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(n_hidden, dim_out)
        )
    
    def forward(self, x):
        return self.mlp(x)
    
    def __len__(self):
        # return the number of parameters
        return sum(p.numel() for p in self.parameters())
    
    def fill(self, params):
        # fill the parameters with a flat tensor
        if len(params) != len(self):
            raise ValueError(f"The number of parameters does not match, expected {len(self)} but got {len(params)}")

        for p in self.parameters():
            n = p.numel()
            p.data.copy_(params[:n].view_as(p))
            params = params[n:]
    
    @classmethod
    def from_parameter(cls, dim_in, dim_out, n_hidden, params, n_hidden_layers=1):
        # create a new instance and fill it with the given parameters
        instance = cls(dim_in, dim_out, n_hidden, n_hidden_layers=n_hidden_layers)
        instance.fill(params)
        return instance


class DiscreteController:
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space
    
    def __call__(self, x):
        with torch.no_grad():
            logits = self.model(x)
            return torch.argmax(logits).item()

class ContinuousController:
    def __init__(self, model, action_space, factor=1):
        self.model = model
        self.action_space = action_space
        self.factor = factor
    
    def __call__(self, x):
        with torch.no_grad():
            result = torch.tanh(self.model(x)).reshape(-1).numpy() * self.factor
            return result