"""Late-entry bridge: the residual stream of a prefix row (a, b) at layer e, predicted from its parent
node's residual stream at layer e (computed in forward 1) and the token embeddings of b and a.

    x_e(a,b) ~= h_e(a) + ridge([h_e(a), E[b], E[a], 1]) + mlp([rms(h_e(a)), E[b], E[a]])

The second forward then runs only layers e..L-1 for the prefix rows, with exact attention to the
prompt and parent K/V. The MLP's output layer starts at zero, so training starts from the ridge fit.
"""
import torch
import torch.nn as nn


class Bridge(nn.Module):
    def __init__(self, D, hidden=2048):
        super().__init__()
        self.D = D
        self.register_buffer('R', torch.zeros(3 * D + 1, D))
        self.inp = nn.Linear(3 * D, hidden)
        self.out = nn.Linear(hidden, D)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer('scale', torch.ones(1))

    def forward(self, ha, eb, ea):
        """ha, eb, ea: [..., D] (any float dtype) -> x_e [..., D] float32."""
        ha, eb, ea = ha.float(), eb.float(), ea.float()
        lin = torch.cat([ha, eb, ea, torch.ones_like(ha[..., :1])], -1) @ self.R
        hn = ha * torch.rsqrt(ha.pow(2).mean(-1, keepdim=True) + 1e-6)
        z = torch.cat([hn, eb * self.scale, ea * self.scale], -1)
        return ha + lin + self.out(torch.nn.functional.silu(self.inp(z)))
