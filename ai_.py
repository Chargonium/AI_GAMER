import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

from math import ceil

torch.manual_seed(39284589)
random.seed(39284589)

# -----------------------------------
# Per-element activation application
# -----------------------------------

def apply_mixed_activation(x, act_ids):
    # act_ids:
    # 0 = linear
    # 1 = sigmoid
    # 2 = tanh
    # 3 = relu
    
    out = torch.empty_like(x)

    mask = (act_ids == 0)
    out[mask] = x[mask]

    mask = (act_ids == 1)
    out[mask] = torch.sigmoid(x[mask])

    mask = (act_ids == 2)
    out[mask] = torch.tanh(x[mask])

    mask = (act_ids == 3)
    out[mask] = F.relu(x[mask])

    return out


# -----------------------------------
# Fully Vectorized Stateful Layer
# -----------------------------------

class FastStatefulLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weights = nn.Parameter(
            torch.rand(in_features, out_features)
        )

        # alpha per neuron
        self.alpha = torch.rand(out_features)

        # state per neuron
        self.register_buffer("state", torch.zeros(out_features))

        # random activation id per neuron
        self.register_buffer(
            "act_ids",
            torch.randint(0, 4, (out_features,))
        )

    def forward(self, x):
        # x: (in_features,)
        weighted = x @ self.weights  # FAST matrix multiply

        # EMA state update (vectorized)
        self.state = (1 - self.alpha) * self.state + self.alpha * weighted

        # Mixed activation
        return apply_mixed_activation(self.state, self.act_ids)


# -----------------------------------
# Full Network
# -----------------------------------

class FastStatefulNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            FastStatefulLayer(426*240*3, 1200),
            FastStatefulLayer(1200, 1000),
            FastStatefulLayer(1000, 800),
            FastStatefulLayer(800, 24),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------
# Test Speed
# -----------------------------------

net = FastStatefulNetwork()

input_data = torch.rand(426*240*3) * 256 - 128
count = 4800

start = time.perf_counter()

for _ in range(count):
    output = net(input_data)

end = time.perf_counter()

duration = end - start
print(f"One pass took: {duration/count * 1000:.3f} milliseconds")
print(output)
