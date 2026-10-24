"""Decode-only gate/up packing; native FP32 linear operations and prefill retained."""
import torch
from torch import nn
from torch.nn import functional as F


class PackedDecodeMLP(nn.Module):
    """Two contiguous weight views share one allocation; packing occurs only at setup.

    Moving/casting this module after installation is unsupported: install after
    model.to(device).eval(), then verify ownership before and after measurement.
    A larger GEMM can change rounding. Mathematical equivalence is not exactness.
    """
    def __init__(self, mlp):
        super().__init__()
        if mlp.training or isinstance(mlp, PackedDecodeMLP):
            raise ValueError('Install once on an evaluation model')
        gate, up = mlp.gate_proj, mlp.up_proj
        if gate.bias is not None or up.bias is not None:
            raise ValueError('Only bias-free gate/up projections are supported')
        if gate.weight.shape != up.weight.shape or gate.weight.dtype != torch.float32 or up.weight.dtype != torch.float32:
            raise ValueError('Matching FP32 gate/up weights required')
        self.intermediate_size, self.hidden_size = gate.weight.shape
        with torch.no_grad():
            packed = torch.cat((gate.weight.detach(), up.weight.detach()), dim=0)
            gate_view, up_view = packed.split(self.intermediate_size, dim=0)
            # Installation must preserve every weight bit, including signed zero.
            assert torch.equal(gate_view.view(torch.int32), gate.weight.detach().view(torch.int32))
            assert torch.equal(up_view.view(torch.int32), up.weight.detach().view(torch.int32))
            gate.weight = nn.Parameter(gate_view, requires_grad=False)
            up.weight = nn.Parameter(up_view, requires_grad=False)
        self.gate_proj, self.up_proj = gate, up
        self.down_proj, self.act_fn = mlp.down_proj, mlp.act_fn
        self.register_buffer('gate_up_weight', packed, persistent=False)
        self.eval()

    def forward(self, x):
        if x.ndim == 3 and x.shape[-2] == 1:
            gate, up = F.linear(x, self.gate_up_weight).split(self.intermediate_size, dim=-1)
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def ownership(self):
        weights = (self.gate_up_weight, self.gate_proj.weight, self.up_proj.weight)
        sizes = (2*self.intermediate_size*self.hidden_size, self.intermediate_size*self.hidden_size)
        assert all(w.dtype == torch.float32 and w.is_contiguous() for w in weights)
        assert self.gate_up_weight.shape == (2*self.intermediate_size, self.hidden_size)
        assert self.gate_proj.weight.shape == self.up_proj.weight.shape == (self.intermediate_size, self.hidden_size)
        assert len({w.untyped_storage().data_ptr() for w in weights}) == 1
        assert [w.storage_offset() for w in weights] == [0, 0, sizes[1]]
        assert all(w.untyped_storage().nbytes() == sizes[0]*4 for w in weights)
        return {'hidden_size':self.hidden_size, 'intermediate_size':self.intermediate_size,
                'unique_weight_bytes':sizes[0]*4, 'original_gate_up_bytes':sizes[0]*4,
                'persistent_extra_weight_bytes':0, 'disjoint_contiguous_views':True}


def install_packed_mlp(model):
    """Replace each Qwen2 MLP after loading; return an independently checkable report."""
    if model.training:
        raise ValueError('Evaluation model required')
    for layer in model.model.layers:
        layer.mlp = PackedDecodeMLP(layer.mlp)
    return packed_mlp_ownership(model)


def packed_mlp_ownership(model):
    layers = [layer.mlp.ownership() for layer in model.model.layers]
    return {'layers':layers, 'weight_bits_preserved_at_install':True,
            'unique_weight_bytes':sum(x['unique_weight_bytes'] for x in layers),
            'persistent_extra_weight_bytes':sum(x['persistent_extra_weight_bytes'] for x in layers)}
