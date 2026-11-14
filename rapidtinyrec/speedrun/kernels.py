"""Triton kernels for the beam decode step.

beam_attention: one program per (user, beam row, kv head). All query heads of
the group and all new query tokens of the row are packed into one tile and
attend jointly to (a) the user's shared prompt K/V, masked to its true length,
and (b) the row's private K/V slots with a causal rule (query t sees private
slots < s0 + t + 1). Softmax in fp32. Replaces ~20 PyTorch ops per layer.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _beam_attn_kernel(Q, KP, VP, KS, VS, O, LEN, scale,
                      sqb, sqr, sqt, sqh,
                      skpb, skph, skpp,
                      sksb, sksr, skss, sksh,
                      sob, sor, sot, soh,
                      R, s0, P,
                      IEEE: tl.constexpr, NQ: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                      BP: tl.constexpr, BS: tl.constexpr, BM: tl.constexpr, KVH: tl.constexpr):
    pid = tl.program_id(0)
    g = pid % KVH
    br = pid // KVH
    r = br % R
    b = br // R
    m = tl.arange(0, BM)                 # packed (query token, head-in-group)
    t = m // REP
    hh = m % REP
    head = g * REP + hh
    mvalid = m < NQ * REP
    d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + r * sqr + t[:, None] * sqt + head[:, None] * sqh + d[None, :],
                mask=mvalid[:, None], other=0.0)
    q = (q.to(tl.float32) * scale).to(Q.dtype.element_ty)
    L = tl.load(LEN + b)
    # shared prefix
    p = tl.arange(0, BP)
    kp = tl.load(KP + b * skpb + g * skph + p[:, None] * skpp + d[None, :], mask=(p < P)[:, None], other=0.0)
    if IEEE:
        sp = tl.dot(q, tl.trans(kp), input_precision="ieee")
    else:
        sp = tl.dot(q, tl.trans(kp))     # [BM, BP] fp32
    sp = tl.where(p[None, :] < L, sp, float('-inf'))
    # private slots
    s = tl.arange(0, BS)
    ks = tl.load(KS + b * sksb + r * sksr + s[:, None] * skss + g * sksh + d[None, :],
                 mask=(s < s0 + NQ)[:, None], other=0.0)
    if IEEE:
        ss = tl.dot(q, tl.trans(ks), input_precision="ieee")
    else:
        ss = tl.dot(q, tl.trans(ks))     # [BM, BS]
    ss = tl.where(s[None, :] <= s0 + t[:, None], ss, float('-inf'))
    mx = tl.maximum(tl.max(sp, 1), tl.max(ss, 1))
    ep = tl.exp(sp - mx[:, None])
    es = tl.exp(ss - mx[:, None])
    den = tl.sum(ep, 1) + tl.sum(es, 1)
    vp = tl.load(VP + b * skpb + g * skph + p[:, None] * skpp + d[None, :], mask=(p < P)[:, None], other=0.0)
    vs = tl.load(VS + b * sksb + r * sksr + s[:, None] * skss + g * sksh + d[None, :],
                 mask=(s < s0 + NQ)[:, None], other=0.0)
    if IEEE:
        acc = tl.dot(ep, vp, input_precision="ieee") + tl.dot(es, vs, input_precision="ieee")
    else:
        acc = tl.dot(ep.to(vp.dtype), vp) + tl.dot(es.to(vs.dtype), vs)
    out = acc / den[:, None]
    tl.store(O + b * sob + r * sor + t[:, None] * sot + head[:, None] * soh + d[None, :],
             out.to(O.dtype.element_ty), mask=mvalid[:, None])


def beam_attention(q, kc, vc, kp, vp, lengths, s0, scale, out=None):
    """q [B,R,Q,H,hd] (unscaled); kc/vc [B,KV,P,hd]; kp/vp [B,R,S,KV,hd]; returns [B,R,Q,H*hd].
    out: optional [B,R,Q,H,hd]-viewable tensor to write into."""
    B, R, NQ, H, D = q.shape
    KV, P = kc.shape[1], kc.shape[2]
    REP = H // KV
    assert H % KV == 0 and kc.stride(3) == 1 and kp.stride(4) == 1 and q.stride(4) == 1
    BP = max(16, triton.next_power_of_2(P))
    BS = 16
    BM = max(16, triton.next_power_of_2(NQ * REP))
    o = torch.empty(B, R, NQ, H, D, device=q.device, dtype=q.dtype) if out is None else out
    _beam_attn_kernel[(B * R * KV,)](
        q, kc, vc, kp, vp, o, lengths, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kc.stride(0), kc.stride(1), kc.stride(2),
        kp.stride(0), kp.stride(1), kp.stride(2), kp.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        R, s0, P, IEEE=q.dtype == torch.float32, NQ=NQ, REP=REP, D=D, BP=BP, BS=BS, BM=BM, KVH=KV, num_warps=4)
    return o if out is not None else o.view(B, R, NQ, H * D)


def reference(q, kc, vc, kp, vp, lengths, s0, scale):
    B, R, NQ, H, D = q.shape
    KV, P = kc.shape[1], kc.shape[2]
    rep = H // KV
    S = s0 + NQ
    qf = q.float() * scale
    kfull = torch.cat([kc.float()[:, None].expand(B, R, KV, P, D), kp[:, :, :S].float().permute(0, 1, 3, 2, 4)], 3)
    vfull = torch.cat([vc.float()[:, None].expand(B, R, KV, P, D), vp[:, :, :S].float().permute(0, 1, 3, 2, 4)], 3)
    kfull = kfull.repeat_interleave(rep, 2)  # [B,R,H,P+S,D]
    vfull = vfull.repeat_interleave(rep, 2)
    sc = torch.einsum('brthd,brhkd->brthk', qf, kfull)
    pidx = torch.arange(P + S, device=q.device)
    tq = torch.arange(NQ, device=q.device)
    valid_p = pidx[None, :] < lengths[:, None]                             # [B,P+S] (prefix part)
    is_priv = pidx >= P
    priv_ok = (pidx[None, :] - P) <= (s0 + tq[:, None])                   # [NQ,P+S]
    ok = torch.where(is_priv[None, None, :], priv_ok[None], valid_p[:, None, :])  # [B,NQ,P+S]
    sc = sc.masked_fill(~ok[:, None, :, None, :], float('-inf'))
    o = torch.einsum('brthk,brhkd->brthd', sc.softmax(-1), vfull)
    return o.reshape(B, R, NQ, H * D)


def torch_path(q, kc, vc, kp, vp, lengths, s0, scale):
    """The engine's PyTorch attention, in the model dtype (same math as Forward.step)."""
    B, R, NQ, H, D = q.shape
    KV, P = kc.shape[1], kc.shape[2]
    rep = H // KV
    S = s0 + NQ
    neg = torch.finfo(torch.float32).min
    pmask = torch.where(torch.arange(P, device=q.device)[None] < lengths[:, None], 0.0, neg)
    qi = torch.arange(NQ, device=q.device)[:, None]
    si = torch.arange(S, device=q.device)[None, :]
    cmask = torch.where(si <= s0 + qi, 0.0, neg)
    q = q * scale
    qg = q.view(B, R, NQ, KV, rep, D).permute(0, 3, 1, 2, 4, 5).reshape(B, KV, R * NQ * rep, D)
    sp = torch.matmul(qg, kc.transpose(-1, -2)).float() + pmask[:, None, None, :]
    sp = sp.view(B, KV, R, NQ * rep, P)
    qr = q.view(B, R, NQ, KV, rep, D).permute(0, 1, 3, 2, 4, 5).reshape(B, R, KV, NQ * rep, D)
    ss = torch.matmul(qr, kp[:, :, :S].permute(0, 1, 3, 4, 2)).float()
    ss = (ss.view(B, R, KV, NQ, rep, S) + cmask[None, None, None, :, None, :]).view(B, R, KV, NQ * rep, S).permute(0, 2, 1, 3, 4)
    probs = torch.softmax(torch.cat([sp, ss], -1), -1).to(q.dtype)
    pp, ps = probs.split([P, S], -1)
    op = torch.matmul(pp.reshape(B, KV, R * NQ * rep, P), vc).view(B, KV, R, NQ * rep, D)
    os_ = torch.matmul(ps, vp[:, :, :S].permute(0, 3, 1, 2, 4))
    return (op + os_).view(B, KV, R, NQ, rep, D).permute(0, 2, 3, 1, 4, 5).reshape(B, R, NQ, H * D)


def selfcheck(dtype, device='cuda'):
    """Max |error| vs an FP32 reference for the Triton kernel and for the PyTorch path in the
    same dtype; the kernel passes if it is no worse than the PyTorch path (+ small slack)."""
    torch.manual_seed(0)
    out = {}
    ok = True
    for (B, R, NQ, s0, P, H, KV) in [(1, 50, 1, 0, 96, 12, 2), (1, 50, 2, 2, 128, 12, 2), (2, 88, 3, 1, 80, 12, 2),
                                     (3, 50, 1, 1, 112, 12, 2), (1, 50, 1, 1, 96, 28, 4), (2, 16, 2, 0, 80, 16, 2)]:
        q = torch.randn(B, R, NQ, H, 128, device=device, dtype=dtype) * 3
        kc = torch.randn(B, KV, P, 128, device=device, dtype=dtype) * 3
        vc = torch.randn_like(kc)
        kp = torch.randn(B, R, 4, KV, 128, device=device, dtype=dtype) * 3
        vp = torch.randn_like(kp)
        lengths = torch.randint(P - 15, P + 1, (B,), device=device)
        r = reference(q, kc, vc, kp, vp, lengths, s0, 128 ** -0.5)
        a = (beam_attention(q, kc, vc, kp, vp, lengths, s0, 128 ** -0.5).float() - r).abs().max().item()
        t = (torch_path(q, kc, vc, kp, vp, lengths, s0, 128 ** -0.5).float() - r).abs().max().item()
        out[f'{B},{R},{NQ},{s0},{P},{H}/{KV}'] = {'triton': a, 'torch': t}
        ok &= a <= 1.5 * t + 1e-3
    out['pass'] = bool(ok)
    return out


# ---------------------------------------------------------------------------------------------
# Prompt rows of the fused first forward: query token t (absolute position T0 + t) attends to the
# user's K/V cache positions <= T0 + t. The cache already holds the shared template (positions
# < T0, written once per buffer) and this layer's new prompt K/V (positions T0..P-1). One program
# per (user, KV head, chunk of CQ query tokens); all query heads of the group packed in one tile.
@triton.jit
def _prompt_attn_kernel(Q, K, V, O, scale,
                        sqb, sqt, sqh, skb, skh, skp, sob, sot, soh,
                        T0, Pn, P, NCH,
                        IEEE: tl.constexpr, CQ: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                        BP: tl.constexpr, BM: tl.constexpr, KVH: tl.constexpr):
    pid = tl.program_id(0)
    c = pid % NCH
    bg = pid // NCH
    g = bg % KVH
    b = bg // KVH
    m = tl.arange(0, BM)
    tl_ = m // REP
    head = g * REP + m % REP
    t = c * CQ + tl_
    mvalid = (tl_ < CQ) & (t < Pn)
    d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + t[:, None] * sqt + head[:, None] * sqh + d[None, :], mask=mvalid[:, None], other=0.0)
    q = (q.to(tl.float32) * scale).to(Q.dtype.element_ty)
    p = tl.arange(0, BP)
    k = tl.load(K + b * skb + g * skh + p[:, None] * skp + d[None, :], mask=(p < P)[:, None], other=0.0)
    if IEEE:
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
    else:
        s = tl.dot(q, tl.trans(k))
    s = tl.where((p[None, :] <= T0 + t[:, None]) & (p[None, :] < P), s, float('-inf'))
    mx = tl.max(s, 1)
    e = tl.exp(s - mx[:, None])
    den = tl.sum(e, 1)
    v = tl.load(V + b * skb + g * skh + p[:, None] * skp + d[None, :], mask=(p < P)[:, None], other=0.0)
    if IEEE:
        acc = tl.dot(e, v, input_precision="ieee")
    else:
        acc = tl.dot(e.to(v.dtype), v)
    tl.store(O + b * sob + t[:, None] * sot + head[:, None] * soh + d[None, :],
             (acc / den[:, None]).to(O.dtype.element_ty), mask=mvalid[:, None])


def prompt_attention(q, kc, vc, T0, scale, out=None, CQ=8):
    """q [B,Pn,H,hd] (unscaled, positions T0..T0+Pn-1); kc/vc [B,KV,P,hd] with P >= T0+Pn.
    Returns [B,Pn,H*hd] (written into `out` if given)."""
    B, Pn, H, D = q.shape
    KV, P = kc.shape[1], kc.shape[2]
    REP = H // KV
    assert H % KV == 0 and kc.stride(3) == 1 and q.stride(3) == 1
    if out is None:
        out = torch.empty(B, Pn, H * D, device=q.device, dtype=q.dtype)
    o4 = out.view(B, Pn, H, D)
    BP = max(16, triton.next_power_of_2(P))
    BM = max(16, triton.next_power_of_2(CQ * REP))
    nch = triton.cdiv(Pn, CQ)
    _prompt_attn_kernel[(B * KV * nch,)](
        q, kc, vc, o4, scale,
        q.stride(0), q.stride(1), q.stride(2), kc.stride(0), kc.stride(1), kc.stride(2),
        o4.stride(0), o4.stride(1), o4.stride(2),
        T0, Pn, P, nch, IEEE=q.dtype == torch.float32, CQ=CQ, REP=REP, D=D, BP=BP, BM=BM, KVH=KV, num_warps=4)
    return out


def prompt_reference(q, kc, vc, T0, scale):
    B, Pn, H, D = q.shape
    KV, P = kc.shape[1], kc.shape[2]
    rep = H // KV
    k = kc.float().repeat_interleave(rep, 1)[:, :, :T0 + Pn]
    v = vc.float().repeat_interleave(rep, 1)[:, :, :T0 + Pn]
    s = torch.einsum('bthd,bhpd->bhtp', q.float() * scale, k)
    ok = torch.arange(T0 + Pn, device=q.device)[None, :] <= (T0 + torch.arange(Pn, device=q.device))[:, None]
    s = s.masked_fill(~ok, float('-inf'))
    return torch.einsum('bhtp,bhpd->bthd', s.softmax(-1), v).reshape(B, Pn, H * D)


def selfcheck_prompt(dtype, device='cuda'):
    torch.manual_seed(0)
    out, ok = {}, True
    for (B, T0, Pn, P, H, KV) in [(1, 69, 7, 80, 12, 2), (1, 69, 52, 128, 12, 2), (3, 69, 20, 96, 12, 2),
                                  (2, 0, 40, 48, 12, 2), (1, 69, 20, 96, 28, 4), (2, 69, 9, 80, 16, 2)]:
        q = torch.randn(B, Pn, H, 128, device=device, dtype=dtype) * 3
        kc = torch.randn(B, KV, P, 128, device=device, dtype=dtype) * 3
        vc = torch.randn_like(kc)
        r = prompt_reference(q, kc, vc, T0, 128 ** -0.5)
        a = (prompt_attention(q, kc, vc, T0, 128 ** -0.5).float() - r).abs().max().item()
        out[f'{B},{T0},{Pn},{P},{H}/{KV}'] = a
        ok &= a < (2e-2 if dtype != torch.float32 else 1e-4)
    out['pass'] = bool(ok)
    return out
