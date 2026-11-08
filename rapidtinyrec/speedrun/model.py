"""Minimal Qwen2 forward with a shared-prefix / private-suffix KV layout.

Weights are loaded directly from the released safetensors file. QKV and
gate/up projections are concatenated so each layer issues four GEMMs.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class Qwen2Weights:
    def __init__(self, ckpt, dtype, device, quant=None):
        ckpt = Path(ckpt)
        cfg = json.loads((ckpt / 'config.json').read_text())
        self.cfg = cfg
        self.L = cfg['num_hidden_layers']
        self.D = cfg['hidden_size']
        self.H = cfg['num_attention_heads']
        self.KV = cfg['num_key_value_heads']
        self.hd = self.D // self.H
        self.I = cfg['intermediate_size']
        self.eps = cfg['rms_norm_eps']
        self.theta = cfg.get('rope_theta', 10000.0)
        self.dtype = dtype
        self.device = device
        sd = load_file(str(ckpt / 'model.safetensors'), device='cpu')

        def w(name):
            return sd[name].to(device=device, dtype=dtype).contiguous()

        self.embed = w('model.embed_tokens.weight')
        self.V = self.embed.shape[0]
        # pad vocabulary rows to a multiple of 128 so the LM-head GEMM gets aligned
        # cuBLAS kernels; padded logits are sliced off before any normalization
        pad = (-self.V) % 128
        if pad:
            self.embed = torch.cat([self.embed, self.embed.new_zeros(pad, self.D)], 0).contiguous()
        if 'lm_head.weight' in sd:
            lm = w('lm_head.weight')
            self.lm_head = torch.cat([lm, lm.new_zeros(pad, self.D)], 0).contiguous() if pad else lm
        else:
            self.lm_head = self.embed
        self.norm = w('model.norm.weight')
        self.layers = []
        for i in range(self.L):
            p = f'model.layers.{i}.'
            a = p + 'self_attn.'
            m = p + 'mlp.'
            self.layers.append(dict(
                ln1=w(p + 'input_layernorm.weight'),
                ln2=w(p + 'post_attention_layernorm.weight'),
                wqkv=torch.cat([w(a + 'q_proj.weight'), w(a + 'k_proj.weight'), w(a + 'v_proj.weight')], 0).contiguous(),
                bqkv=torch.cat([w(a + 'q_proj.bias'), w(a + 'k_proj.bias'), w(a + 'v_proj.bias')], 0).contiguous(),
                wo=w(a + 'o_proj.weight'),
                wgu=torch.cat([w(m + 'gate_proj.weight'), w(m + 'up_proj.weight')], 0).contiguous(),
                wd=w(m + 'down_proj.weight'),
            ))
        del sd
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.int64).float() / self.hd))
        t = torch.arange(4096, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], -1)
        self.cos = emb.cos().to(device=device, dtype=dtype)
        self.sin = emb.sin().to(device=device, dtype=dtype)
        self.linear = F.linear
        if quant is not None:
            quant(self)

    def param_bytes(self):
        seen = set(); total = 0
        items = [self.embed, self.lm_head, self.norm] + [v for l in self.layers for v in l.values()]
        tensors = []
        for t in items:
            if isinstance(t, torch.Tensor):
                tensors.append(t)
            elif isinstance(t, torch.nn.Module):
                tensors += list(t.parameters()) + list(t.buffers())
                tensors += [v for v in vars(t).values() if isinstance(v, torch.Tensor)]
        for t in tensors:
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr()); total += t.numel() * t.element_size()
        return total


def rmsnorm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope(x, cos, sin):
    # x: [..., n_heads, hd]; cos/sin broadcast as [..., 1, hd]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


class Forward:
    """Layer math shared by prefill and beam steps."""

    def __init__(self, W, attn='torch'):
        self.W = W
        self.scale = 1.0 / math.sqrt(W.hd)
        self.attn = attn
        self.layer_fn = None   # optional compiled whole decode layer (see compile_layer)

    def decode_layer(self, x, pos, kc, vc, kp, vp, lengths, s0: int, layer):
        """One decode layer with Triton beam attention; a single torch.compile region."""
        from .kernels import beam_attention
        q, k, v = Forward.qkv(self, x, layer, pos)
        Q = x.shape[2]
        kp[:, :, s0:s0 + Q] = k
        vp[:, :, s0:s0 + Q] = v
        o = beam_attention(q, kc, vc, kp, vp, lengths, s0, self.scale)
        return Forward.post(self, x, o, layer)

    def compile_layer(self, mode='default'):
        self.layer_fn = torch.compile(self.decode_layer, mode=mode, dynamic=False)

    def qkv(self, x, layer, pos):
        W = self.W
        h = rmsnorm(x, layer['ln1'], W.eps)
        qkv = W.linear(h, layer['wqkv'], layer['bqkv'])
        q, k, v = qkv.split([W.H * W.hd, W.KV * W.hd, W.KV * W.hd], -1)
        shp = x.shape[:-1]
        q = q.reshape(*shp, W.H, W.hd)
        k = k.reshape(*shp, W.KV, W.hd)
        v = v.reshape(*shp, W.KV, W.hd)
        cos = W.cos[pos].unsqueeze(-2)
        sin = W.sin[pos].unsqueeze(-2)
        return rope(q, cos, sin), rope(k, cos, sin), v

    def post(self, x, o, layer):
        W = self.W
        x = x + W.linear(o, layer['wo'])
        h = rmsnorm(x, layer['ln2'], W.eps)
        gu = W.linear(h, layer['wgu'])
        g, u = gu.split([W.I, W.I], -1)
        return x + W.linear(F.silu(g) * u, layer['wd'])

    def final(self, x):
        return rmsnorm(x, self.W.norm, self.W.eps)

    # -- trie-level prefill fusion: prompt [B,P,D] plus n first-level trie nodes [B,n,D] in ONE
    # forward. Prompt tokens are causal; node t attends to the user's real prompt (< L) and to
    # itself only, at position L. Prompt K/V -> kcache/vcache; node K/V -> private slot 0.
    # Optional shared template: if `tmpl` = (kt [L,KV,T0,hd], vt) holds the K/V of a prefix
    # shared by every prompt, `x` covers only prompt positions T0..P-1 and attends to the
    # cached template as well (exact: causal attention never lets the template see later tokens).
    def prefill_tree(self, x, xn, lengths, kcache, vcache, kpriv, vpriv, tmpl=None, capture=None):
        W = self.W
        B, Pn, _ = x.shape
        T0 = 0 if tmpl is None else tmpl[0].shape[2]
        P = T0 + Pn
        n = xn.shape[1]
        T = Pn + n
        rep = W.H // W.KV
        dev = x.device
        pos = torch.cat([torch.arange(T0, P, device=dev)[None].expand(B, Pn), lengths[:, None].expand(B, n)], 1)
        qi = torch.arange(T, device=dev)                 # query index within new tokens
        kj = torch.arange(T0 + T, device=dev)            # key index: template, new prompt, nodes
        causal = kj[None, :] <= (qi[:, None] + T0)                                    # [T,T0+T]
        node_q = qi[:, None] >= Pn
        to_prompt = (kj[None, None, :] < lengths[:, None, None]) & (kj[None, None, :] < P)
        self_ok = kj[None, :] == (qi[:, None] + T0)
        mask = torch.where(node_q[None], to_prompt | self_ok[None], causal[None])     # [B,T,T0+T]
        h = torch.cat([x, xn], 1)
        for li, layer in enumerate(W.layers):
            if capture is not None and li in capture:
                capture[li] = h[:, Pn:].clone()      # node residual stream entering layer li
            q, k, v = self.qkv(h, layer, pos)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            if T0:
                kfull = torch.cat([tmpl[0][li][None].expand(B, -1, -1, -1), kt], 2)
                vfull = torch.cat([tmpl[1][li][None].expand(B, -1, -1, -1), vt], 2)
            else:
                kfull, vfull = kt, vt
            kcache[li].copy_(kfull[:, :, :P])
            vcache[li].copy_(vfull[:, :, :P])
            kpriv[li][:, :, 0] = k[:, Pn:]
            vpriv[li][:, :, 0] = v[:, Pn:]
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), kfull.repeat_interleave(rep, 1), vfull.repeat_interleave(rep, 1),
                attn_mask=mask[:, None], scale=self.scale)
            h = self.post(h, o.transpose(1, 2).reshape(B, T, W.H * W.hd), layer)
        h = self.final(h)
        return h[:, :Pn], h[:, Pn:]

    @torch.inference_mode()
    def template_kv(self, ids, device):
        """K/V of a prompt prefix shared by all requests, computed once at setup."""
        W = self.W
        t = torch.tensor([ids], device=device)
        T0 = len(ids)
        kc = torch.zeros(W.L, 1, W.KV, T0, W.hd, device=device, dtype=W.dtype)
        vc = torch.zeros_like(kc)
        self.prefill(W.embed[t], torch.arange(T0, device=device)[None], kc, vc)
        return kc[:, 0].contiguous(), vc[:, 0].contiguous()

    # -- prefill: x [B, P, D], pos [B, P]; returns hidden and per-layer K/V [B, KV, P, hd]
    def prefill(self, x, pos, kcache, vcache):
        W = self.W
        B, P, _ = x.shape
        rep = W.H // W.KV
        for i, layer in enumerate(W.layers):
            q, k, v = self.qkv(x, layer, pos)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            kcache[i].copy_(kt)
            vcache[i].copy_(vt)
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), kt.repeat_interleave(rep, 1), vt.repeat_interleave(rep, 1),
                is_causal=True, scale=self.scale)
            x = self.post(x, o.transpose(1, 2).reshape(B, P, W.H * W.hd), layer)
        return self.final(x)

    # -- beam step: x [B, R, Q, D] (R beams per user, Q new tokens per beam)
    # prefix K/V: [B, KV, P, hd] with additive prefix mask [B, P]
    # private K/V: [B, R, S, KV, hd]; the new tokens occupy slots [s0, s0+Q)
    def step(self, x, pos, kcache, vcache, pmask, kpriv, vpriv, s0, lengths=None, n_layers=None, start=0,
             capture=None):
        """start > 0: x is the residual stream entering layer `start` (late entry); layers below are
        skipped for these rows. capture: dict {layer: None} filled with the input of those layers."""
        W = self.W
        B, R, Q, _ = x.shape
        rep = W.H // W.KV
        S = s0 + Q
        P = kcache[0].shape[2]
        neg = torch.finfo(torch.float32).min
        # causal mask among private slots: query q sees slots < s0 + q + 1
        qi = torch.arange(Q, device=x.device)[:, None]
        si = torch.arange(S, device=x.device)[None, :]
        cmask = torch.where(si <= s0 + qi, 0.0, neg)  # [Q, S]
        if self.attn == 'triton' and self.layer_fn is not None:
            for i, layer in enumerate(W.layers[:n_layers]):
                if i < start:
                    continue
                x = self.layer_fn(x, pos, kcache[i], vcache[i], kpriv[i], vpriv[i], lengths, s0, layer)
            return self.final(x)
        for i, layer in enumerate(W.layers[:n_layers]):
            if i < start:
                continue
            if capture is not None and i in capture:
                capture[i] = x.clone()
            q, k, v = self.qkv(x, layer, pos)  # q [B,R,Q,H,hd], k/v [B,R,Q,KV,hd]
            kpriv[i][:, :, s0:S] = k
            vpriv[i][:, :, s0:S] = v
            if self.attn == 'triton':
                from .kernels import beam_attention
                o = beam_attention(q, kcache[i], vcache[i], kpriv[i], vpriv[i], lengths, s0, self.scale)
                x = self.post(x, o, layer)
                continue
            q = q * self.scale  # pre-scale (as SDPA does) to keep fp16 logits in range
            # group query heads by kv head: [B, KV, R*Q*rep, hd]
            qg = q.view(B, R, Q, W.KV, rep, W.hd).permute(0, 3, 1, 2, 4, 5).reshape(B, W.KV, R * Q * rep, W.hd)
            sp = torch.matmul(qg, kcache[i].transpose(-1, -2)).float()  # [B,KV,RQr,P]
            sp = sp + pmask[:, None, None, :]
            sp = sp.view(B, W.KV, R, Q * rep, P)
            # private scores: [B, R, KV, Q*rep, S]
            qr = q.view(B, R, Q, W.KV, rep, W.hd).permute(0, 1, 3, 2, 4, 5).reshape(B, R, W.KV, Q * rep, W.hd)
            kp = kpriv[i][:, :, :S].permute(0, 1, 3, 4, 2)  # [B,R,KV,hd,S]
            ss = torch.matmul(qr, kp).float()  # [B,R,KV,Q*rep,S]
            ss = ss.view(B, R, W.KV, Q, rep, S) + cmask[None, None, None, :, None, :]
            ss = ss.view(B, R, W.KV, Q * rep, S).permute(0, 2, 1, 3, 4)  # [B,KV,R,Qr,S]
            scores = torch.cat([sp, ss], -1)
            probs = torch.softmax(scores, -1).to(x.dtype)
            pp, ps = probs.split([P, S], -1)
            op = torch.matmul(pp.reshape(B, W.KV, R * Q * rep, P), vcache[i]).view(B, W.KV, R, Q * rep, W.hd)
            vp = vpriv[i][:, :, :S].permute(0, 3, 1, 2, 4)  # [B,KV,R,S,hd]
            os_ = torch.matmul(ps, vp)  # [B,KV,R,Qr,hd]
            o = (op + os_).view(B, W.KV, R, Q, rep, W.hd).permute(0, 2, 3, 1, 4, 5).reshape(B, R, Q, W.H * W.hd)
            x = self.post(x, o, layer)
        return self.final(x)
