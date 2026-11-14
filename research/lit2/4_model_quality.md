# 4. Model quality for SID generative recommendation: literature digest

Survey date: 2026-11-08. Scope: the model-quality side of MiniOneRec-style generative recommendation (GR) with LLMs and semantic IDs (SIDs):
- follow-ups on Amazon Office / Industrial
- training objectives aligned with decoding
- SID work that applies to a fixed SID set
- test-time compute
- training aimed at efficiency

Goal: choose cheap fine-tunes (≤1 H100-hour per run, about 10 runs) and inference-only ideas for the released MiniOneRec checkpoint under our exact decoding.

**Verification legend**
- **[V]**: numbers or mechanism read this session from the paper's arXiv HTML or abstract page, a GitHub README, or a Hugging Face card.
- **[S]**: from search-engine snippets or secondary summaries only. Treat as unverified.
- **[R]**: from our own repo (`research/PLAN.md`, `research/tune_*.md`).

All HR/NDCG values are on a 0–1 scale unless marked %. With one positive per user, HR@K = Recall@K.

---

## 0. Bottom line

1. **Our certified exact decode already sits mid-pack among published 2026 *trained* improvements on Office.** It runs on the unmodified released checkpoint.
   - Office [R]: HR@10 0.1712 / NDCG@10 0.1319. The MiniOneRec beam-50 reproduction is 0.1589 / 0.1247.
   - For comparison:
     - S-GRec: 0.1689 / 0.1308
     - V-STAR: 0.1746 / 0.1340
     - SA-OPSD: 0.1747 / 0.1334
     - DASO (1.5B): HR@10 0.1834
   - On Industrial, decoding-side gains are null [R]: about 0.159–0.161 HR@10, about 0.117–0.118 NDCG@10. Model changes are the only lever there. Published Industrial ceilings are SA-OPSD 0.1690 / 0.1263 and ReRe-Base 0.1707 / 0.1256 (title-based).
2. **Compare our numbers with caution.** Most follow-ups copy MiniOneRec's paper numbers (Office 0.1634 / 0.1242, Industrial 0.1586 / 0.1167) rather than rerunning. Our beam-50 reproduction of the released checkpoint is lower on Office HR@10 (0.1589). CAPE's frozen released checkpoint gives 0.1445 HR@10 at beam 50. Leaderboard deltas are therefore not on a common baseline.
3. **The largest, best-documented gains come from fixing GRPO's dead groups.** When none of the 16 beam rollouts contains the target, the group gives zero gradient. Three fixes:
   - **DASO**: prefix-guided replacement rollouts, an SID-prefix reward m/M, and an SFT anchor with λ=0.1. Office HR@10 0.1634 → 0.1834 at 1.5B.
   - **Sibling-GRPO / V-STAR**: advantages normalized within each set of sibling prefixes. Office HR@10 → 0.1684 with beam decoding, 0.1746 with VED.
   - **HCGRec**: a hinted prefix trained with SFT plus GRPO on the suffix only.

   All three start from the SFT checkpoint and run full RL on 8 GPUs. With our exact engine as the rollout generator, a shortened version fits about one H100-hour.
4. **The best inference-only, exact-compatible idea is collaborative score fusion.**
   - Sources: D3's text-free assistant, TCBSA, ISD, CAPE.
   - CAPE on the *same released checkpoint and test set*: Office HR@10 +0.82 pt and NDCG@10 +0.36 pt; Industrial −0.11 pt.
   - D3's per-token SASRec conditional is ≤ 0, so our admissible-bound certificate survives.
5. **Decoding-aligned objectives exist for beam, but none for exact search.**
   - Beam-aligned: BEAR (per-step top-B margin), APAO (prefix-aware), SA-OPSD (search-aware self-distillation).
   - Restricted softmax: MSL (trie-masked softmax with adaptive temperature).
   - No paper trains under the *trie-renormalized* objective we decode with, or measures the training/decoding normalization mismatch for SID GR. That gap is open and cheap to test.
6. **Quality ∩ speed.**
   - A LayerSkip-style "late-entry consistency" fine-tune would make the third (and first) code tolerate deeper late entry.
   - Margin-sharpening losses (BEAR, temperature < 1, prefix reward) would raise certificate rates at smaller M.
   - Training on certified top-K rollouts would make the trained policy equal the decoded one.
   - Evidence is indirect: LayerSkip, ILRec cross-layer KL, SLMRec, IBA, our logit lens.

---

## 1. Office / Industrial leaderboard

### 1.1 Protocol facts

- **MiniOneRec protocol**
  - Amazon Reviews 2018; 5-core; history truncated to 10 items; chronological 8:1:1 split.
  - SIDs: 3-level RQ-VAE, 256 codes per level, built on Qwen3-Embedding-4B.
  - Constrained beam search without length normalization [V].
  - The paper gives no evaluation beam width. DASO and CAPE both use beam 50.
- **Test sizes match ours.** Office has 4,866 test cases (CAPE [V]) and Industrial 4,533 (CAPE, MiniOneRec-study, community HF cards [V]). These match our 4,866 / 4,533.
- **Copied baselines.**
  - The MiniOneRec rows in S-GRec, UniVA, V-STAR, DASO and SA-OPSD are the paper's numbers, not reruns.
  - D3 and S-DPO rows on Industrial are identical to ReRe's Table 1, so they are copied from ReRe.
- **Inconsistencies in source papers.**
  - MiniOneRec Table 3 Industrial (HR@10 0.1546, NDCG@10 0.1139) ≠ Table 1 (0.1586 / 0.1167).
  - S-DPO Office NDCG@10 is 0.1255 in MiniOneRec but 0.1188 in SA-OPSD.
  - UniVA's public-benchmark rows are identical to S-GRec's. Both are Tencent papers sharing authors, so this is one result reported twice.

### 1.2 Office Products (HR@10 / NDCG@10)

"Same split" means the MiniOneRec 8:1:1 protocol. "Same SIDs" means the official MiniOneRec SID assignment.

**Our runs [R]**

| Method | HR@10 | NDCG@10 | Backbone / decode | Same split | Same SIDs | Notes |
|---|---|---|---|---|---|---|
| MiniOneRec beam-50, FP32 (our reproduction) | 0.1589 | 0.1247 | Qwen2-1.5B released ckpt | Y | Y | the paired reference |
| Certified exact, trie scoring (M=16) | 0.1652 | — | same | Y | Y | inference-only |
| Exact, codebook(a,b)+trie(c), w_c=0.75, + late entry | **0.1712** | **0.1319** | same | Y | Y | ΔHR@10 +1.23 [+0.66, +1.83] vs reference |

**Same split, same model family**

| Method | HR@10 | NDCG@10 | Backbone / decode | Same split | Same SIDs | Notes |
|---|---|---|---|---|---|---|
| MiniOneRec (paper) [V] | 0.1634 | 0.1242 | Qwen2.5-1.5B (DASO labels it 1.5B; one HF card says 7B) | Y | Y | the number every follow-up copies |
| D3 (as reported by MiniOneRec) [V] | 0.1634 | 0.1213 | text-based LLM | Y | n/a | |
| S-DPO (as reported by MiniOneRec) [V] | 0.1587 | 0.1255 | text-based | Y | n/a | SA-OPSD lists NDCG 0.1188 |
| BIGRec / TIGER / LC-Rec / HSTU / SASRec [V] | 0.1434 / 0.1408 / 0.1237 / 0.1400 / 0.1120 | 0.1091 / 0.1002 / 0.0920 / 0.1126 / 0.0858 | | Y | — | MiniOneRec Table 1 |
| **S-GRec** (2602.10606) [V] | 0.1689 | 0.1308 | Qwen2.5-1.5B, init from MiniOneRec SFT ckpt | Y | Y | A2PO with an LLM semantic judge (Qwen3-4B) |
| UniVA (2605.05803) [V] | 0.1689 | 0.1308 | — | Y (stated) | ? | identical to S-GRec |
| **V-STAR**, VED decoding (2602.10699) [V] | 0.1746 | 0.1340 | Qwen2.5-1.5B | Y | likely | Sibling-GRPO + value-guided decoding |
| V-STAR, beam decoding (same model) [V] | 0.1684 | 0.1299 | — | Y | likely | isolates the training gain |
| **DASO 1.5B** (2608.20611) [V] | **0.1834** | n/r (NDCG@5 0.1344) | Qwen2.5-1.5B-Instruct, beam 50 | Y | **Y (official)** | single run; Office 1.5B was used to pick B_max ([S] Pith review) |
| DASO 3B [V] | 0.1893 | n/r (NDCG@5 0.1331) | Qwen2.5-3B | Y | Y | |
| Sibling-GRPO (rerun in DASO) [V] | 0.1746 | n/r | 1.5B | Y | Y | equals V-STAR's VED number |
| MiniOneRec + GT in group (DASO) [V] | 0.1683 | n/r | 1.5B | Y | Y | add the target to the rollout group |
| SFT-only (DASO) [V] | 0.1431 | n/r (NDCG@5 0.0931) | 1.5B | Y | Y | RL is worth ≈ +2 pt here |
| **SA-OPSD** (2611.01568) [V] | 0.1747 | 0.1334 | Qwen2.5-1.5B-Instruct | Y | ? | privileged self-teacher; 5.79 GPU-h |
| APAO (rerun in SA-OPSD) [V] | 0.1710 | 0.1262 | 1.5B | Y | ? | prefix-aware loss |
| SmartGR (rerun in SA-OPSD) [V] | 0.1697 | 0.1295 | 1.5B | Y | ? | hierarchy- and beam-aware KD |
| CAPE, frozen released ckpt (2610.29298) [V] | 0.1445 → **0.1527** | 0.1120 → 0.1156 | Qwen2.5-1.5B released ckpt, beam 50 + 25 co-occurrence items | Y | Y | training-free; note the low backbone HR |

**Different split or different SIDs**

| Method | HR@10 | NDCG@10 | Backbone / decode | Same split | Same SIDs | Notes |
|---|---|---|---|---|---|---|
| UGR (2602.11719, KDD'26) [V] | 0.1598 | 0.1231 | Qwen3-8B LoRA, 4-level SIDs, beam 10 | ~ (8:1:1, own counts: 4,861 test) | N | MiniOneRec rerun 0.1479 / 0.1137; SFT-only 0.1563 / 0.1213 > GRPO 0/1 reward 0.1485 / 0.1120 |
| SIDReasoner (2603.23183) [V] | 0.1648 | 0.1208 | Qwen3-1.7B, own multi-domain SIDs | ~ | N | reason then answer; GRPO with a prefix-match reward |
| Evo-Rec (2609.29973) [V] | 0.1710 | 0.1299 | Qwen3-1.7B, beam 10 | ~ | N | best-of-N reasoning SFT + NDCG@10 reward |
| TS-Rec, SFT only (2602.22632) [V] | 0.1541 | 0.1175 | Qwen2.5-1.5B, beam 20 | ~ | ? | SID-token semantic init + token-level alignment |
| MuonRec 1.5B (2603.00416) [V] | 0.1560 (Adam 0.1476) | 0.1231 (Adam 0.1192) | MiniOneRec-1.5B | **N** (leave-one-out, 2,024 users) | ? | Muon optimizer, 25% fewer steps |
| ChronoID (2606.14260) [V] | 0.1359 (MiniOneRec 0.0953) | 0.0895 | — | **N** (time cutoff) | N | not comparable |
| TCBSA + MiniOneRec (2610.32098) [V] | R@10 0.0424 → 0.0436 | 0.0262 → 0.0268 | — | **N** | ? | different protocol; relative gain only |

### 1.3 Industrial & Scientific (HR@10 / NDCG@10)

**Our runs [R]**

| Method | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| MiniOneRec beam-50 (our reproduction) | 0.1582 | ≈0.117 | |
| Certified trie M=16 / + late entry | 0.1588 / 0.1597 | ≈0.117 | quality-neutral, 2.6–2.7× speed |
| Full-depth reference used in the bridge runs | 0.1613 | 0.1181 | scoring-rule gains are n.s. on Industrial |

**Same split, same model family**

| Method | HR@10 | NDCG@10 | Backbone / decode | Same split | Same SIDs | Notes |
|---|---|---|---|---|---|---|
| MiniOneRec (paper) [V] | 0.1586 | 0.1167 | 1.5B | Y | Y | Table 3 says 0.1546 / 0.1139 |
| D3 / S-DPO (from ReRe Table 1) [V] | 0.1500 / 0.1524 | 0.1082 / 0.1082 | text-based | Y | n/a | |
| SPRec (ReRe) [V] | 0.1525 | 0.1139 | text-based | Y | n/a | |
| **ReRe-Base** (2510.12211) [V] | **0.1707** | **0.1256** | Qwen2-0.5B, **title-based** constrained beam | Y | n/a (titles) | GRPO with beam rollouts + rank reward; not SIDs |
| ReRe-SFT [V] | 0.1546 | 0.1113 | same | Y | n/a | |
| S-GRec / UniVA [V] | 0.1632 | 0.1202 | 1.5B | Y | Y | |
| V-STAR, VED / beam [V] | 0.1641 / 0.1606 | 0.1217 / 0.1194 | 1.5B | Y | likely | |
| DASO 1.5B / 3B [V] | 0.1632 / 0.1667 | n/r (NDCG@5 0.1183 / 0.1112) | | Y | Y (official) | MiniOneRec 3B 0.1673 beats DASO 3B |
| **SA-OPSD** [V] | **0.1690** | **0.1263** | 1.5B | Y | ? | best SID-based Industrial number found |
| APAO / SmartGR (reruns in SA-OPSD) [V] | 0.1652 / 0.1648 | 0.1202 / 0.1229 | 1.5B | Y | ? | |
| CAPE, frozen released ckpt [V] | 0.1257 → 0.1246 | 0.0866 → 0.0863 | released ckpt, beam 50 | Y | Y | tail HR@10 +3.4 pt, overall flat |

**Different split or different SIDs**

| Method | HR@10 | NDCG@10 | Backbone / decode | Same split | Same SIDs | Notes |
|---|---|---|---|---|---|---|
| UGR [V] | 0.1393 (MiniOneRec rerun 0.1276) | 0.1033 (0.0980) | Qwen3-8B, 4-level SIDs | ~ | N | |
| SIDReasoner / Evo-Rec [V] | 0.1438 / 0.1476 | 0.1010 / 0.1076 | Qwen3-1.7B | ~ | N | |
| TS-Rec, SFT only [V] | 0.1462 | 0.1029 | 1.5B | ~ | ? | |
| MuonRec 1.5B [V] | 0.1460 (Adam 0.1372) | 0.1092 (0.1048) | — | **N** (leave-one-out) | ? | |
| ChronoID [V] | 0.1622 (MiniOneRec 0.1353) | 0.1241 | — | **N** | N | |

**Community results (self-reported cards) [V-card]**

| Model | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| onesfour MiniOneRec-1.5B, SFT | 0.1337 | 0.0945 | beam 50; well below the paper |
| onesfour MiniOneRec-1.5B, GRPO | 0.1379 | 0.1005 | GDPO ≈ SFT |
| Liquor-06 two-stage SASRec distilled from Qwen3-1.7B | 0.165 (3 seeds 0.164) | 0.119 | **0.8M parameters**; own tuned SASRec 0.134 / 0.104, far above the MiniOneRec table's SASRec (0.1088 / 0.0806) |
| continuejpg MiniOneRec-study, Qwen2.5-0.5B | HR@20 0.198 (SFT) → 0.162–0.176 (GRPO) | — | beam 20; **every GRPO variant was worse than SFT**; shuffling SIDs costs −45% HR@20 |

### 1.4 Reading the leaderboard

- **Office.**
  - Published trained gains over the copied 0.1634 baseline: +0.5 pt (S-GRec), +1.1 pt (V-STAR / SA-OPSD) and +2.0 pt (DASO).
  - Our decode-only +1.2 pt over our own 0.1589 reproduction is the same order as the RL and distillation papers. The protocol is not identical, but the comparison is useful.
  - A stacked "exact decode + DASO-style RL" result would be new.
- **Industrial.**
  - Decoding changes are neutral [R]. Trained methods give +0.5–1.0 pt (S-GRec, DASO, V-STAR, SA-OPSD).
  - The title-based ReRe reaches 0.1707 with a 0.5B model, so the SID bottleneck may matter more here.
  - A distilled 0.8M-parameter SASRec hits 0.165: the item space is tiny (3.7k items), and collaborative signal is undertapped.
- **GRPO versus SFT is contested at larger K.**
  - UGR's own ablation: SFT 0.1563 > GRPO 0.1485 HR@10 on Office.
  - MiniOneRec-study: GRPO < SFT at HR@20.
  - DASO: SFT 0.1431 → MiniOneRec RL 0.1634.
  - The effect depends on reward and anchoring. Every successful variant keeps an NLL/SFT anchor: DASO λ_sft=0.1; HCGRec prefix SFT; AdaGRPO "GRPO+NLL"; GenRec (JD) GRPO-SR.

---

## 2. MiniOneRec and its Office/Industrial follow-ups: paper cards

**MiniOneRec** (Kong, Sheng, Tan, Chen, Wu, Zhang, Wang, He; arXiv 2510.24431, 2025-10-28) [V]
- **SFT.** SID tokens are added to Qwen2.5-Instruct. Tasks:
  - next-SID from SID history
  - text history → SID, and SID history → title
  - SID ↔ title alignment
  - description reconstruction
  - preference summarization (DeepSeek pseudo-labels)

  Up to 10 epochs at lr 3e-4.
- **RL.** GRPO for 2 epochs at lr 1e-5, KL unchanged.
  - **Rollouts** are constrained beam search of width 16 (distinct and valid), without length normalization.
  - **Reward** is 1 for the target plus a rank penalty on negatives: −1/log(ρ+1), group-normalized.
  - A **SASRec collaborative reward** caused reward hacking.
  - **Dynamic sampling** was dropped.
- **Ablations.** Alignment helps. Pretrained weights ≫ scratch (Office HR@10 0.1634 vs 0.1196).
- **Repo since release** [V README]:
  - RQ-Kmeans, RQ-Kmeans+ and constrained RQ-Kmeans SIDs
  - an option to freeze the LLM and train only the SID embeddings
  - VAFT/HEPO (GPR-inspired) SFT and RL variants
  - TS-Rec, FLASH, PRQ-KMeans, RQ-OPQ
  - a "MiniOneRec-Think" roadmap item
  - a warning that a non-zero CC metric means constrained decoding failed with Instruct models

**ReRe**: Reinforced Preference Optimization for Recommendation (Tan et al., Alibaba/NUS; 2510.12211) [V]
- MiniOneRec's RL precursor, with *title-based* items.
- Constrained beam rollouts versus sampling (Industrial H@10 / N@10):
  - beam 0.1705 / 0.1220
  - dynamic 0.1540 / 0.1122
  - common 0.1447 / 0.1044
- Ranking reward versus rule-only reward: N@10 0.1256 vs 0.1220.
- Semantic and collaborative rewards collapse (H@10 0.06–0.09).
- GRPO ≈ DAPO ≈ GSPO.
- Length normalization was removed, following D3.
- Cost: 8×H20, about 4 h per dataset; G=16; lr 1e-5; β=1e-3; batch 512.

**UGR**: Uncertainty-aware GR (Fan, Gao et al., USTC; 2602.11719, KDD'26) [V]
- Reward: incorrect beam candidates are penalized in proportion to their softmax share of log-probability ("UW-reward").
- Each context's GRPO loss is weighted by the target's rank (difficulty).
- Confidence tokens [C_HIGH]/[C_LOW] are trained with a pairwise loss; re-ranking with confidence adds +0.14 pt NDCG@10.
- Ablation on Office HR@10:
  - SFT 0.1563
  - GRPO 0/1 reward 0.1485
  - rank reward 0.1468
  - UW-reward 0.1522
  - + difficulty 0.1579
  - full 0.1598
- Lesson: naive GRPO hurt HR@10 relative to SFT, and a better negative-weighting reward recovers it.

**S-GRec** (Tencent; 2602.10606) [V]
- Starts from the MiniOneRec SFT checkpoint and MiniOneRec splits.
- A Qwen3-4B personalized semantic judge gives a semantic advantage. It is added only when its sign agrees with the business-proxy advantage, scaled by the min/max magnitude ratio (A2PO).
- Variants (Office HR@10 / NDCG@10):
  - naive reward-sum collapses (0.0904 / 0.0761)
  - advantage-sum 0.1515 / 0.1223
  - full 0.1689 / 0.1308
- Querying the judge on 5% of instances keeps 99%+ of the gain.

**V-STAR**: Spend Search Where It Pays (Tencent; 2602.10699) [V]
- **VED** (value-guided decoding): a value head (1 Transformer block + MLP) trained by TD on semantic-similarity step rewards. A UCB-style expansion with an entropy bonus runs under a token budget.
- **Sibling-GRPO**: z-normalized advantages within sibling sets under each parent prefix, combined with global GRPO.
- Same model, beam vs VED (Office HR@10 / NDCG@10): 0.1684 / 0.1299 vs 0.1746 / 0.1340.
- The authors report that 1× VED matches 3–4× beam on Office.
- Online: +1.23% GMV on WeChat Channels.

**DASO**: Difficulty-Aware SID Optimization (Meta/PSU; 2608.20611) [V]
- Uses the official MiniOneRec SIDs unchanged.
- Per prompt, it profiles prefix-match depth of the rollout group and replaces up to B_max=9 of the weakest rollouts with prefix-guided completions. Forced prefix tokens are excluded from the ratio and KL.
- Adds an SID-prefix reward R_sid = m/M (weight 1.0) and an SFT anchor λ_sft=0.1.
- GRPO settings: G=16, β=1e-3, lr 1e-5, **600 updates**, 8×A100; eval with 50-beam constrained decoding.
- Ablations at 1.5B (Office HR@10):
  - no prefix reward 0.179
  - B_max 5 / 9 / 13: 0.179 / 0.1834 / 0.181
  - no SFT anchor: HR@5 0.159 vs 0.1639
- Also reports the target is often outside the top-16 of the 50-beam ranking at the SFT checkpoint. That is the "dead group" motivation.

**HCGRec**: Hint-Conditioned RL (SJTU/Meituan; 2608.11980, CIKM'26) [V]
- Datasets: Instruments, Arts, Games (leave-one-out). No Office/Industrial results.
- The hint is the shortest target SID prefix at which a diagnostic rollout from the SFT model hits the target.
- Loss: prefix SFT (λ=0.005) + suffix GRPO.
- Zero-gradient groups fall from >70% to <20%.
- Instruments HR@10 0.1180 vs MiniOneRec 0.1145; HR@50 0.1985 vs 0.1696. The large-K gain is the notable part.

**SA-OPSD**: Search-Aware On-Policy Self-Distillation with Sibling Alignment (Kuaishou/UvA; 2611.01568, WWW'27 submission) [V]
- The teacher is the same 1.5B model given the target's title (privileged). It scores the student's own beam-search states.
- States are selected near the pruning boundary ("EBS").
- The KL is split into a target-mass term and a sibling-relation term.
- Cost: 5.79 GPU-h, versus 17.77 for dense full-KL.
- Ablation (NDCG@10, Industrial / Office):
  - uniform state selection 0.1216 / —
  - EBS 0.1263 / 0.1334
  - plain OPD 0.1168 / 0.1224

**CAPE** (USTC/Ant; 2610.29298) [V]
- Training-free, on the released MiniOneRec checkpoints.
- Adds 25 tail items chosen by user-level co-occurrence to the beam-50 pool. All are scored by teacher forcing.
- Fusion: normalized generator log-likelihood + λ·(tail-only co-occurrence evidence). The top-1 item is anchored.
- Office HR@10 +0.82 pt, NDCG@10 +0.36 pt; Industrial −0.11 pt overall, tail +3.4 pt. Latency +11–16%.

**AdaGRPO** (JD; 2606.08480) [V]
- Proprietary data.
- On a proprietary dataset, GRPO alone lowered HR@10 (11.01 → 10.39). GRPO+NLL gave 11.06, and per-sample gating of GRPO by difficulty and reward reliability gave 11.63.
- Supports always-on NLL plus selective GRPO.

**MuonRec** (SJTU/Kuaishou; 2603.00416) [V]
- Muon on 2-D hidden weights (Adam elsewhere) gives about 25% fewer steps to converge and better final metrics. Office 1.5B: HR@10 0.1476 → 0.1560, leave-one-out protocol.
- Most of the gain comes from using Muon in SFT (Muon-SFT / Adam-RL ≈ Muon / Muon).

**TS-Rec** (MiniOneRec group; 2602.22632) [V]
- SA-Init: SID token embeddings initialized from LLM-extracted keywords of each token's item cluster.
- TS-Align: token ↔ description tasks.
- SFT-only Office HR@10 0.1541. As an SFT backbone for MiniOneRec RL, Industrial gains are +8.7–15.7%.
- It is not shown as an add-on to an existing checkpoint.

**SIDReasoner** (2603.23183) and **Evo-Rec** (2609.29973) [V]
- Both use Qwen3-1.7B and reason before emitting the SID.
  - SIDReasoner: GRPO with a stepwise prefix-match reward plus a format reward (λ=0.1).
  - Evo-Rec: best-of-5 teacher traces kept by likelihood gain on the target SID, then GRPO with an NDCG@10 reward over beam-10 ranks.
- Office HR@10 0.1648 / 0.1710. Neither compares to MiniOneRec or reports latency.

**Retrieval-grounded credit assignment** (2609.29983) [V abstract]
- Same group as Evo-Rec.
- Reasoning traces list interest hypotheses. A frozen retriever checks each one, giving per-hypothesis credit; the SID span is untouched.
- Abstract only; no numbers.

**ChronoID** (2606.14260) [V]
- Time-aware SIDs (relative-time sinusoid, parallel codebooks) on a time-cutoff split. Not comparable.

**UniVA** (2605.05803) [V]
- Ads value alignment. Its public table duplicates S-GRec's.

---

## 3. Training objectives aligned with decoding

### 3.1 Trie- and constraint-aware softmax during training

**MSL**: "Not All Tokens Are What You Need for Tuning LLM as a Recommender" (Wang et al., SIGIR'25; 2504.04178) [V]
- Loss: CE whose denominator includes only *valid* next tokens, using a trie over item titles.
- The restricted denominator causes vanishing gradients. The fix is an **Adaptive Temperature Strategy**: τ is set in closed form from the mean and variance of training logits, the mean valid-token count m, and a target probability η=0.25.
- Without temperature, MSL is *worse* than plain CE on Toy and Office.
- BIGRec + MSL NDCG@10 vs CE: Office 0.0220 → 0.0438; Toy 0.0182 → 0.0288. Average +42% over CE.
- About 4–7% faster than CE; S-DPO costs about 4× MSL.
- The backbone was text-based LLaMA3-8B. There is **no SID experiment**.
- **Relevance:** the closest training-time counterpart of our trie-renormalized scoring. For 3-level SIDs, level-3 child sets are tiny (often 1–3 items). Without τ or reweighting, MSL's level-3 term carries almost no gradient.

**Sibling-relative signals**
- Sibling-GRPO (V-STAR) and SA-OPSD's sibling-relation term are the RL and distillation versions of "compete only with valid siblings". Both gain on our splits (§2).

**SToICaL** (Google/UMass; 2601.05588) [V]
- Token targets spread over *valid* trie children in proportion to the ranks of relevant items beneath them, plus item-level reweighting λ(r)=1/r^α.
- Shows trie marginalization improves NDCG and Recall@k>1 but can cut R@1 (ESCI R@1 95 → 70).
- Not recommendation (WordNet, ESCI); Mistral-7B.

**"Constrained Auto-Regressive Decoding Constrains Generative Retrieval"** (Wu, Ren, de Rijke et al., SIGIR'25; 2504.09935) [V abstract]
- Theory: adding corpus constraints to a Bayes-optimal model yields error lower-bounded by the KL between true and model step-wise marginals.
- Beam search over marginals is a poor basis for top-k recall when relevance is sparse.
- This is the formal reason trie-renormalization changes rankings.

**Open gap.** No paper measures the effect of training normalization (full-vocabulary CE as MiniOneRec does) against decoding normalization (trie-renormalized, codebook-restricted, or full-vocabulary plus mask) on SID GR ranking.
- Our scoring study is the decode half of that experiment.
- The training half is a cheap fine-tune (§7, recipe C).

### 3.2 Beam-aware and search-aware training

**BEAR**: Beam-Search-Aware Optimization (Yang, Chen et al., ZJU; 2601.22925, SIGIR'26) [V]
- Loss: L = L_SFT + λ Σ_t log σ_ξ(log β_t^B − log p(y_t|y_<t)), where β_t^B is the B-th largest next-token probability at step t.
- This is a per-step necessary condition for the positive to survive beam pruning.
- Uses logits already computed in SFT, so cost ≈ SFT. The exact-prefix version costs 4.45×.
- Office (text-based BIGRec, Llama-3.2-3B, LoRA r=8, lr 1e-4, 10 epochs, B=10): N@10 0.0421 vs MSL 0.0388, S-DPO 0.0292, D3 0.0237. Average +12.5% over the best baseline.
- Grid: λ ∈ [0.1, 5], ξ ∈ [0.25, 3].

**APAO**: Adaptive Prefix-Aware Optimization (Tsinghua; 2603.02730, KDD'26) [V]
- Loss: L_CE + β Σ_m w_m L_m, where L_m is a prefix-level loss:
  - pointwise: CE on the first m tokens
  - pairwise: positive versus 100 random negative prefixes of the same length
- Weights w_m ∝ w_m·exp(η L_m), so the worst prefix gets the most weight. Use β ∈ [0.1, 0.4].
- Early-prefix supervision matters most.
- Rerun on MiniOneRec splits by SA-OPSD: Office 0.1710 / 0.1262, Industrial 0.1652 / 0.1202. Both are above the copied MiniOneRec baseline.

**PAG / planning-ahead** (SIGIR'24, 2404.14600) and **"Lost in Decoding?"** reproduction (2604.23396, SIGIR'26) [S]
- A set-based look-ahead prior reduces early pruning of relevant prefixes, but it is fragile when the plan collapses.

**Gryphon** (Yandex; 2606.08604) [V]
- A joint next-SID loss plus an item-level scorer (sampled softmax with logQ correction) that re-scores beam candidates.
- Item re-scoring beats the SID-level beam ceiling (Recall@1000 0.8552 vs 0.8404). The authors read this as beam-likelihood *miscalibration*.
- Private data only.

**BARGE** (Tencent; 2607.21028) [V]
- A per-level contrastive path scorer (HPR) reranks an expanded pool: score = log p + 0.25·log softmax(r_l).
- TIGER-scale on Beauty/Sports, +16.6% R@10.

**PROMISE** (Kuaishou; 2601.04674) [V]
- A path-level process reward model (InfoNCE per depth) guides beam search.
- Industrial Recall@100 +48–59%. On public Sports, Recall@5 0.0450 vs 0.0316.
- The PRM is trained jointly; it is not a cheap add-on.

### 3.3 Listwise and preference losses for LLM recommenders

**S-DPO** (NeurIPS'24, 2406.09215) [S]
- A softmax DPO with multiple negatives, linked to sampled-softmax hard-negative mining.
- MiniOneRec Office 0.1587 / 0.1255 (as reported).
- Costs about 2–10× BEAR or MSL.

**SPRec** (WWW'25, 2412.09243) [S]
- Self-play loop of SFT then DPO, using the previous iteration's outputs as negatives; debiases popularity.

**ILRec** (RUC/Meituan; 2602.17410) [V]
- Self-hard negatives come from the ensemble logits of the last k intermediate layers.
- Cross-layer preference optimization reweights the softmax denominator.
- **Cross-layer KL distillation** pulls intermediate distributions toward the final layer.
- A SASRec soft-label regularizer guards against false negatives.
- On LC-Rec (which uses SIDs), Instruments HR@10 0.1062 → 0.1143.
- Faster than S-DPO and SPRec (4.2 h vs 7.25 h on Instruments).
- Relevant to both quality and late entry (§8).

**LETTER ranking-guided loss** (CIKM'24, 2405.07314) [V]
- Temperature-scaled CE, with τ dividing the logits. The authors say a smaller τ penalizes hard negatives more and link this to one-way partial AUC.
- The gain is small (TIGER Instruments R@10 0.1092 → 0.1122) and drops when τ is too small. The sweep covers 0.6–1.2.

**Token-weighted objectives** (2601.17787, CIKM'26) [V]
- Front-greater per-level weights (by dispersion reduction per level) plus frequency weights, with a curriculum.
- TIGER Industrial & Scientific: +3.7% H@5, +1.3% H@10.

### 3.4 Level and length normalization

**D3**: Decoding Matters (Bao et al., EMNLP'24; 2406.14900) [V]
- "Ghost tokens" (p≈1) make length normalization inflate some items, so D3 *removes* length normalization.
- A **text-free assistant** (SASRec) gives per-token scores:

  log Σ_{eligible items with next token t} p_S / Σ_{items matching prefix} p_S

  The combination is S̃ = α·S_LLM + (1−α)·S_TF.
- Qwen1.5-1.8B HR@10 gains of +4–26% across six Amazon sets. Without the assistant: −0.2 to −0.75 pt.
- On TIGER, the SID version also gains (Games 0.0651 → 0.0706).
- MiniOneRec and ReRe inherit "no length normalization" from D3.

**Our own result [R]**
- Codebook(a,b) + trie(c) with w_c (0.5–0.75 tuned on validation) is a per-level weight. It gives +1.1–1.2 pt HR@10 on Office and nothing on Industrial.
- No paper studies per-level temperature or weights for SID decoding (§9).

---

## 4. SID construction, as it bears on a fixed SID set

These papers change the SIDs, so most are out of scope. What carries over to a fixed SID set:

- **LETTER** (2405.07314) [V]: collaborative + diversity regularized RQ-VAE. Its transferable piece is the ranking-guided τ-CE loss (§3.3), which is small.
- **ETEGRec** (2409.05546) [S]: joint tokenizer and recommender training with sequence-item and preference-semantic alignment. This needs re-tokenization, so not applicable.
- **TS-Rec** (2602.22632) [V]: SID token embeddings are the weak point. SA-Init plus token-level alignment add +8–14% at SFT.
  - Fixed-SID add-on: a few hundred steps of TS-Align tasks (token ↔ cluster description).
  - Untested on a trained checkpoint.
  - The MiniOneRec repo can freeze the LLM and train only SID embeddings, which is a cheap probe.
- **Expressiveness limits / Latte** (Hou, Ju, Shah, McAuley; 2605.06331) [V]
  - Items that share a long prefix get coupled probabilities (correlation ≈ 1.0 at tree distance 2).
  - Rank reversals between siblings are hard to express.
  - Latte prepends 4–8 latent tokens, giving multiple trees: +3.45% NDCG@10.
  - Needs retraining; it explains why sibling-level losses help.
- **IIRG**, memorization in LLM GR (Snap/KAIST; 2606.17276) [V]
  - LLM gains over TIGER come mostly from one-hop memorized transitions.
  - Adding collaborative-neighbor and semantic-neighbor generation tasks gives +21% Recall@5 on average, and +40% for users not covered by one-hop memorization.
  - Qwen 4B; Sports, Toys, Beauty.
  - Works with a fixed SID set as extra SFT tasks.
- **Model-scaling view** (2509.25522, KDD'26) [V abstract]: SID-based GR saturates with scale, and SID capacity is the bottleneck. LLM-as-RS scales better (up to +20%).
- **SID reproducibility study** (2609.24430) [V]: no SID scheme is universally best. Larger backbones and longer SIDs do not reliably help.
- **MiniOneRec-study** [V]: shuffling item→SID assignments costs −42 to −46% HR@20. The current SIDs carry real structure.
- **Churn and freezing** [S]: DACT (2603.29705), DREAM (2606.06947), SIDScope. Frozen SIDs avoid code churn but place new items poorly. Irrelevant for a fixed catalog.

> **Not found.** The exact titles "Coupling SID construction and history supervision" and "Freezing SIDs under item churn" did not match any paper. Closest: Masked History Learning (MHL, 2509.23649, ACL'26) [V abstract] adds masked-history reconstruction to the GR objective, with entropy-guided masking and a curriculum. It is a fixed-SID add-on with no reported numbers seen. Churn papers are listed above.

---

## 5. Test-time compute for GR

**Explicit reasoning before the SID**
- Methods: SIDReasoner, Evo-Rec, OneRec-Think (2510.11639; "Think-Ahead" for deployment) [V abstract], R2ec (NeurIPS'25) [S], CogRec (2607.24402) [V], TwiSTAR (2605.11553) [S].
- Gains on Office are similar to or below the RL-only methods. Neither SIDReasoner nor Evo-Rec reports latency.
- CogRec decodes 5 traces × beam 10.
- Evo-Rec reports that weak reasoning can *hurt*.
- TwiSTAR routes about 62% of users to the fast path [S].
- **For us:** this means hundreds of extra tokens per user, which wipes out the 3 ms budget. Exact decoding still applies to the final SID conditioned on the trace, but the search runs per trace.

**Latent reasoning**
- Methods: ReaRec (2503.22675) [S], LARES [S], IBA "Where Reasoning Matters" (2607.12425) [V], BRIDGE (2611.07425) [V abstract].
- IBA finds information gain falls with SID position. Positions 1–2 benefit from extra refinement steps; positions 3–4 are near saturation.
- A (3,2,1,0) allocation beats uniform at lower latency.
- BRIDGE spends more recurrence on early levels.
- **For us:** this matches our logit lens: codes are decided late, and the third code needs few layers. Extra compute belongs on level 1.

**Value- or reward-guided search**
- V-STAR VED (Office +0.6 pt HR@10 over beam on the same model) and PROMISE PRM.
- Both rank by a *learned value*, not the model's log-probability. Under exact decoding they act as rerankers over a certified top-N pool (two-stage).

**Collaborative test-time fusion** (training-free)
- **D3 TFA**: per-token SASRec conditional (§3.4).
- **TCBSA** (2610.32098) [V]: SASRec margins above the R-th score (R=500) are aggregated into prefix→token edge potentials (top-8 mean). They are added as η·A(p,t), with confidence gate η = λσ(k(m−τ)). Office settings: λ=0.8, k=13, τ=0.10. On MiniOneRec: Office R@10 +2.8% relative, Yelp +12%. Latency 1.4–1.6× (beam 20).
- **ISD** (2607.24995) [V]: reciprocal-rank fusion of decoder rank and an item ranker's rank at each prefix. TIGER Office R@10 0.1194 → 0.1367. Beam-10 retains only 35% of top-ranked targets after the third token.
- **CAPE**: §2.
- **Exactness.**
  - D3's per-token SASRec term is log of a conditional probability, so it is ≤ 0. Any fused score α·Σ log p_LLM + β·Σ log q_SAS with α, β ≥ 0 keeps our "all increments ≤ 0" admissible bound. Certification survives.
  - TCBSA's potentials are nonnegative bonuses, so the bound needs + λ·max-potential-under-prefix.
  - CAPE is a rerank of a pool.

**Test-time augmentation (TTA)**
- TTA4SR (Dang et al., SIGIR'25; 2504.04843) [S]: TMask and TNoise on the history, averaged. Beats training-time augmentation on SASRec and GRU4Rec.
- AdaTTA (2604.16121) [S]: picks the operator per user with RL.
- No GR or LLM-SID study was found.
- **For us:** averaging trie log-probabilities over K masked histories is exact-compatible (a mean of ≤0 terms), at K× prefill.

**Multi-prompt and self-consistency**
- **No GR paper found.** MiniOneRec trains both SID-history→SID and text-history→SID tasks. Summing their per-step trie log-probabilities is a product-of-experts that stays exact-compatible at 2× prefill.
- Indirect evidence:
  - Llama4Rec-style LLM+CF ensembles [S]
  - ILRec (averaging the last-k-layer logits helps as a negative source)

**Temperature**
- Under trie-renormalized scoring, per-level temperature T_l changes the item ranking. Under full-vocabulary log-probabilities with no renormalization it changes it too.
- No GR study was found. Our w_c tuning is the nearest evidence.

---

## 6. Training aimed at efficiency

**LayerSkip** (Meta, ACL'24; 2404.16710) [V]
- **Layer dropout** p_{l,t} = S(t)·D(l)·p_max, with D(l) = e^{l ln2/(L−1)} − 1 (rising exponentially toward the last layer).
- **Early-exit loss** through the *shared* LM head, with weights ẽ(t,l) ∝ C(t,l)·e(l), where e(l) = e_scale·Σ_{i≤l} i (quadratic in depth).
- **Curriculum**: rotational (every R-th layer) or gradual.
- **Recipes**:
  - finetune TOPv2 (Llama 1.5B, 24 layers): p_max=0.2, e_scale=1.0, gradual curriculum
  - finetune code: p_max=0.1, e_scale=1.0, R=16
  - continual pretraining: p_max=0.1, e_scale 0.1–0.2, R=8
- **Effects**:
  - TOPv2 layer-12 exact match goes from 0 to 77%, at about −3% on the last layer.
  - Llama3 continual pretraining drops more (MMLU 66.5 → 60.5).
  - Speedups 1.8–2.2×.

**SLMRec** (ICLR'25; 2405.17890) [V]
- Teacher E4SRec (LLaMA-7B, 8 decoder layers kept); student 4 layers.
- Losses: CE + (1−cos) + 0.1·L2 on block-boundary hidden states (4 blocks), plus per-block supervised heads with λ3 between 0.01 and 1.
- The student *beats* the 8-layer teacher and matches the 32-layer model, at 6.6× training and 8× inference speed.
- Finding: intermediate LLM layers are redundant for recommendation. Truncated hidden states fail *without* training.

**SmartGR** (ZJU/Ant; 2608.02048) [V]
- OneRec-8B → OneRec-1.7B distillation.
- Losses:
  - level-weighted KL on the teacher beam with the longest shared prefix (learned tanh depth weights)
  - pairwise beam-rank KL (adjacent teacher beams)
  - total L = L_hard + 0.6·L_SID + 0.3·L_BEAM
- +8.6% average over the student; lr 5e-6, 4 epochs, 3.2 h on 8×A100 for Beauty.
- Rerun on our splits (SA-OPSD): Office 0.1697 / 0.1295.

**SID-MLP** (2605.12617) [V]
- A TIGER teacher distilled into per-digit MLP heads (KL + CE).
- 8.7–10× throughput with NDCG@10 equal or better.
- Already in our inference ledger.

**AtSpeed** (ICLR'25; 2410.05165) [S]
- Draft-model alignment for top-K beam speculative decoding.
- AtSpeed-S minimizes reverse KL plus a density regularizer: about 2× under strict top-K verification, up to 2.5× relaxed.

**Reassessing Layer Pruning** (ICLR'26) [V abstract]
- Pruning the *last* layers and fine-tuning lm_head + the last 3 remaining layers beats sophisticated metrics and LoRA healing.

**ILRec cross-layer KL** (§3.3) [V]: makes intermediate layers' next-token distributions track the final layer while *improving* final quality. This is the quality-positive version of an early-exit loss.

**HERO** (2611.03272) [V]: on-policy distillation from head and tail teachers into a student over its own beam prefixes. Hit@20 beats BEAR, GKD and SID-GRPO on Beauty and Pet. Small BART-MoE models.

**AR vs multi-token prediction (MTP)** (Netflix-style catalog; 2610.15864, NeurIPS'26 workshop) [V abstract]
- Single-head autoregressive beam beats MTP on ranking at about 2.6× lower training cost for Qwen3-8B. MTP wins only on throughput.
- FLASH (2610.07402) [V] pairs SimHash SIDs with parallel heads; its benchmarks are not ours.

---

## 7. Ranked recipes for our setting

Constraints:
- the released MiniOneRec Qwen2-1.5B checkpoint (RL-trained)
- Office about 39k training sequences (8:1:1); validation about 4.9k
- ≤1 H100-hour per run, about 10 runs
- evaluation by certified exact decode with validation-tuned scoring

Ranking is by expected HR@10 / NDCG@10 gain per GPU-hour. Expected ranges are my estimates from the cited evidence, not measurements.

### A. Exact-compatible collaborative fusion at inference (0 GPU-h; expected Office +0.3–1.0 pt HR@10, Industrial ±0.3)

1. **Train the assistant.** Train SASRec on the *training split* item sequences: 2 blocks, d=64, max length 10, full-softmax CE, 200 epochs, early-stop on validation NDCG. This takes minutes. Card [V] shows a tuned SASRec reaching 0.134 HR@10 on Industrial, versus 0.109 in the MiniOneRec table.
2. **Precompute per-user conditionals.** Over the trie, compute
   - q(a) = Σ_{i∈a} p_S(i)
   - q(b|a) = Σ_{i∈ab} p_S / Σ_{i∈a} p_S
   - q(c|ab), likewise

   This is one sparse scatter-add over 3.4k items per user (D3 Eq. 4).
3. **Score.** s(i) = Σ_l w_l log p̃_LLM(c_l | c_<l) + β Σ_l log q(c_l | c_<l). Optionally add a TCBSA confidence gate β = β0·σ(13(m − 0.10)), where m is SASRec's max softmax probability.
   - Every increment is ≤ 0, so the bound and certificate are unchanged.
   - Our forward-2 node selection uses lp(a)+lp(b|a), which becomes the fused version.
4. **Tune.** Grid β0 ∈ {0.05, 0.1, 0.2, 0.3, 0.5} jointly with the existing w_c on validation, then report paired bootstraps on test.
5. **Variant.** CAPE-style: add tail items by co-occurrence to the certified top-50 pool and rerank. Office +0.82 pt on this checkpoint.

**Evidence:** D3 assistant ablation (+0.2–0.75 pt HR@10, LLM 1.8B), TCBSA on MiniOneRec (+2.8% relative on Office), ISD (+1.4–1.7 pt R@10 on TIGER Office), CAPE (+0.82 pt on the released checkpoint).

**Risk:** MiniOneRec found a SASRec *reward* gets hacked in RL. Inference fusion has no such loop.

### B. Short GRPO continuation with exact rollouts, prefix reward, SFT anchor and sibling advantages (≈1 GPU-h; expected +0.5–1.5 pt HR@10 on both datasets)

- **Start point.** The MiniOneRec SFT checkpoint if obtainable; S-GRec used it, and all published deltas are measured from SFT. Otherwise use the released RL checkpoint and expect smaller gains.
- **Rollouts.** Our certified exact top-16 under the *same trie-renormalized, w-weighted scoring we decode with*: distinct, valid, and no search error. About 1,000+ users/s on H100, so a full-epoch rollout takes about 40 s.
  - If the target is not in the top-16, replace the weakest B_max=9 with target-prefix-guided completions (DASO), excluding forced tokens from the ratio and KL.
  - Cheaper alternative: insert the ground truth as one candidate ("MiniOneRec+GT": +0.5 pt on Office).
- **Reward.**
  - 1[hit]
  - − 0.3·(rank penalty −1/log2(ρ+1), group-normalized)  (MiniOneRec λ_rank)
  - + 1.0·(m/3), the SID-prefix depth reward (DASO)
- **Advantage.** Global group z-score + Sibling-GRPO z-score within each parent prefix (V-STAR "Joint").
- **Loss.** PPO-clip ε=0.2, KL β=1e-3 to the start checkpoint, + λ_sft=0.1·CE(ground-truth SID) as the anchor. Log-probabilities in the ratio are trie-renormalized so training matches decoding.
- **Optimization.** AdamW, lr 1e-5 (MiniOneRec/DASO), 64–128 prompts per update, 300–600 updates (0.5–1 epoch), bf16, single H100.
  - Use a shared-prompt tree mask (prompt + 16×4 suffix tokens), as in our engine. Otherwise the 16× expanded batch costs about 80M tokens per epoch, which is about 1 GPU-h.
- **Evidence:**
  - DASO 1.5B Office 0.1634 → 0.1834 HR@10, Industrial → 0.1632 (from SFT; single runs)
  - V-STAR beam-decoded Office 0.1684, Industrial 0.1606
  - HCGRec HR@50 +2.9 pt on Instruments
- **Monitor.** HR@50/100 as well as HR@10. UGR and MiniOneRec-study show GRPO can lose recall at larger K. Keep the anchor.

### C. "Trie-consistent" SFT continuation: MSL on SIDs + BEAR margin (≈0.3 GPU-h; expected 0 to +1 pt, high uncertainty, low cost)

- **Data.** Office next-SID task (about 39k), optionally plus 20% of MiniOneRec's alignment tasks under ordinary CE.
- **Loss per level l.**
  - Masked CE over the valid children of c_<l. Level 1 = the 88 valid first codes on Office (48 on Industrial).
  - Temperature τ from MSL's ATS formula or a grid {0.7, 1.0}.
  - Level weights (1, 1, 0.5), because level-3 child sets are tiny.
- **Plus a BEAR margin term.** λ_B Σ_l log σ_ξ(log β_l^M − log p̃(c_l|c_<l)), with β_l^M = the M-th largest valid-child probability. Use M=16 at level 1, matching our certified frontier, and M=4 at level 2. Grid λ_B ∈ {0.1, 0.5}, ξ=1.
- **Anchor.** + 0.1·KL(p̃_ref ‖ p̃_θ) on trie-renormalized distributions, against the frozen released model.
- **Optimization.** lr 1e-5, cosine, 1 epoch, batch 128, full-parameter bf16 (about 24 GB of optimizer state; fits). About 6M tokens, 10–20 min.
- **Evidence:**
  - MSL +42% NDCG (text items; needs adaptive temperature)
  - BEAR +12.5% over S-DPO and MSL (text items) at SFT cost
  - APAO on our splits Office 0.1710 / 0.1262
  - SA-OPSD's sibling term (−0.3 pt NDCG@10 without it)
- **Why it may be small.** Trie renormalization at inference already removes probability mass on invalid tokens. The gain must come from re-allocating gradient to sibling competition.
- **Bonus.** Raises certificate rates at small M (§8).

### D. Multi-view and TTA ensembling at inference (0 GPU-h; 2–4× prefill; expected 0 to +0.8 pt; speculative)

- Sum the trie log-probabilities from:
  - the SID-history prompt
  - the title-history prompt (MiniOneRec's "text history → SID" task)
  - optionally 2 TMask-style histories (drop 1 random history item)
- Weights tuned on validation. All terms are ≤ 0, so the result is exact-compatible.
- **Evidence:** TTA4SR and AdaTTA on SASRec [S]; general ensembling. No GR result exists; this would be new.
- It costs extra prefill, so try it only if A–C stall.

### E. Privileged self-distillation, a cheap SA-OPSD (≈1 GPU-h if trimmed; expected +0.5–1 pt)

- **Teacher.** The same frozen model with the target's title appended to the prompt.
- **Student states.** Our certified top-16 prefixes, keeping only states near the certificate boundary: the K-th score minus the bound is small.
- **Loss.** L_SID + λ_KD·KL(teacher ‖ student) at level-1 and level-2 states. Weight the sibling-relation part by max(1 − q_y, 0.1).
- SA-OPSD costs 5.8 GPU-h in full; cut states 5× to fit.
- **Evidence:** SA-OPSD Office 0.1747 / 0.1334, Industrial 0.1690 / 0.1263 (best SID-based Industrial number found).

**Runners-up, not ranked**
- IIRG neighbor-generation tasks: large gains, but on 4B models and other datasets; extra SFT on an RL checkpoint risks undoing RL.
- TS-Align SID-token tasks, or SID-embedding-only training: cheap probe; the repo supports freezing the LLM.
- Muon for any SFT stage: −25% steps.
- LETTER τ-CE: small gain.

---

## 8. Ideas that improve both quality and speed

1. **Late-entry consistency fine-tune (LayerSkip for our late entry).**
   - Today: entry at layer 18 is lossless and at 26 costs −0.45 pt HR@1. First-level late entry at layer 18 is lossless only with cascade-trained bridges [R].
   - Proposal: fine-tune the top layers + the bridge jointly. For level-2 and level-3 positions, sample entry e ∈ {18, 22, 24, 26} (rotational curriculum, R=4) and add γ·KL(full ‖ late-entry) + γ'·CE(late). Keep a KL-to-frozen anchor on the full-depth path. Use γ=0.3, lr 5e-6, and train only layers ≥ 14, so ≤0.3 GPU-h.
   - Evidence:
     - LayerSkip TOPv2 (mid-layer exit 0 → 77% exact match, −3% last layer)
     - ILRec (cross-layer KL is quality-*positive*)
     - SLMRec (4–8 layers ≈ 32 when distilled)
     - "Reassessing layer pruning" (heal with lm_head + last 3 layers)
     - IBA (later SID positions saturate)
   - Payoff: deeper lossless entry (6 → 2–4 layers for c, and possibly b) means fewer layer-rows per forward. Batched throughput at B=32 already rose 1.66× from first-level late entry.
2. **Margin-sharpening objectives give cheaper certificates.**
   - Sources: BEAR per-level top-M margin, the DASO prefix reward, LETTER τ<1, RL sharpening generally.
   - The certificate fires when the K-th exact score ≥ the best unexpanded bound. Larger gaps between in-frontier and out-of-frontier prefixes mean smaller M (rows in forward 2/3) at the same certification rate. Today: cert@10 0.54 at N=128 under full-vocabulary scoring; 100% at M=16 under trie scoring.
   - Training BEAR with M equal to our frontier size directly targets "the target's prefix is inside the top-M".
   - Caveat: sharpening can reduce large-K recall (MiniOneRec-study, CAPE's long-tail forgetting), which hurts the certified top-500 story. Measure HR@500.
3. **Train on certified rollouts (B with the trie-renormalized ratio).** Rollouts become the decoded list and the objective becomes the scoring rule. This removes the train/decode mismatch and should increase top-K stability, the certificate rate at small M, and HR@K together.
4. **A restricted LM head becomes "free" once training is restricted.**
   - If quality is defined by trie- or codebook-restricted probabilities (C), the 151,936-row tied LM head (about 233M parameters, ≈15% of Qwen2-1.5B weight bytes) is never needed at decode, and no full-vocabulary semantics need preserving.
   - The FlashRec precedent does restricted log-softmax.
5. **Smaller students via SID-aware KD.**
   - Options: SmartGR level-weighted KL + beam-rank KL into Qwen2.5-0.5B; SLMRec-style 1.5B → ~10-layer student with block cosine/L2 + intermediate CE.
   - Evidence: MuonRec shows a 0.5B MiniOneRec at 0.1560 Office HR@10 (leave-one-out protocol). Training may exceed 1 GPU-h.
   - Offers 2–3× bytes per forward. The most speed, and the most risk.
6. **Search-aware self-distillation.** Make beam-W ≈ exact for small W (SA-OPSD, V-STAR). This matters only if the engine falls back to beam. Lower priority given certified search.

---

## 9. Gaps, contradictions and unverified items

- **Open (no paper found):**
  - training-time vs decode-time normalization mismatch for SID GR
  - per-level temperature or weights at decode
  - multi-prompt or self-consistency ensembling for SID GR
  - LayerSkip or early-exit *training* for SID recommenders
  - RL with exact (certified) rollouts
  - any MiniOneRec follow-up reporting latency together with quality on Office/Industrial
- **Contradictions:**
  - The paper's MiniOneRec Office 0.1634 vs our released-checkpoint reproduction 0.1589 vs CAPE's 0.1445 (all nominally beam-constrained on the released or paper model). Report only paired deltas against our own reproduction.
  - DASO's 1.5B MiniOneRec row equals the paper's numbers. The paper also hints at 7B (HF card says "MiniOneRec (7B, paper)").
  - S-DPO Office NDCG@10: 0.1255 vs 0.1188.
  - GRPO helps (DASO, MiniOneRec) vs hurts (UGR at 8B LoRA, MiniOneRec-study at 0.5B) relative to SFT.
- **Unverified [S]:**
  - S-DPO / SPRec / ETEGRec / PAG / "Lost in Decoding" / TTA4SR / AdaTTA / ReaRec / LARES / R2ec / TwiSTAR / AtSpeed / DREAM / DACT details
  - the OneRec-Think mechanism beyond its abstract
  - the Pith note that DASO tuned B_max on Office 1.5B
  - RecZero (MiniOneRec README cites it, NeurIPS'25; not fetched)
  - CogRec / IBA / BRIDGE / Evo-Rec latency (not reported)

## 10. Sources (arXiv IDs; all fetched via arxiv.org/abs or /html unless marked [S])

**MiniOneRec and Office/Industrial follow-ups**
- MiniOneRec 2510.24431, plus github.com/AkaliKong/MiniOneRec
- ReRe 2510.12211
- UGR 2602.11719
- S-GRec 2602.10606
- UniVA 2605.05803
- V-STAR 2602.10699
- DASO 2608.20611
- HCGRec 2608.11980
- SA-OPSD 2611.01568
- CAPE 2610.29298
- AdaGRPO 2606.08480
- MuonRec 2603.00416
- TS-Rec 2602.22632
- SIDReasoner 2603.23183
- Evo-Rec 2609.29973
- Retrieval-grounded credit 2609.29983
- ChronoID 2606.14260
- TCBSA 2610.32098
- SOTA2 Office leaderboard page

**Community reproductions and model cards**
- github.com/continuejpg/MiniOneRec-study
- huggingface.co/onesfour/MiniOneRec-1.5B-SFT-GDPO
- huggingface.co/Liquor-06/SASRec-TwoStage-Industrial-distillops

**Training objectives aligned with decoding**
- MSL 2504.04178
- BEAR 2601.22925
- APAO 2603.02730
- SToICaL 2601.05588
- Constrained AR decoding 2504.09935
- D3 2406.14900
- LETTER 2405.07314
- Token-weighted objectives 2601.17787
- ILRec 2602.17410
- S-DPO 2406.09215 [S]
- SPRec 2412.09243 [S]
- PAG 2404.14600 [S]
- Lost in Decoding 2604.23396 [S]
- Gryphon 2606.08604
- BARGE 2607.21028
- PROMISE 2601.04674

**Item selection at decode time**
- ISD 2607.24995

**SID construction and analysis**
- Latte 2605.06331
- IIRG 2606.17276
- Model-scaling view 2509.25522
- SID reproducibility study 2609.24430
- GRID 2507.22224
- MHL 2509.23649
- ETEGRec 2409.05546 [S]
- CapsID 2605.05096
- UniGRec 2601.17438
- FLASH 2610.07402
- PrefixMem 2606.00324
- EchoRec 2608.14011
- HoloRec 2606.15331
- CogRec 2607.24402
- IBA 2607.12425
- BRIDGE 2611.07425
- UAGR 2611.02981
- HERO 2611.03272

**Test-time compute and reasoning**
- OneRec-Think 2510.11639
- R2ec 2505.16994 [S]
- ReaRec 2503.22675 [S]
- TwiSTAR 2605.11553 [S]
- TTA4SR 2504.04843 [S]
- AdaTTA 2604.16121 [S]

**Efficiency-motivated training**
- LayerSkip 2404.16710
- SLMRec 2405.17890
- SmartGR 2608.02048
- SID-MLP 2605.12617
- AtSpeed 2410.05165 [S]
- Reassessing Layer Pruning (ICLR'26)
- AR vs MTP 2610.15864

**Other**
- GenRec (JD) 2604.14878
- OpenOneRec 2512.24762 [S]
