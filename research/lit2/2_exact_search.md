# Lit sweep 2: exact, optimal and certified decoding; generative-retrieval decoding guarantees

Sweep date: 2026-11-08. Scope: exact and optimal search for autoregressive models, A* and k-best search, optimality certificates, generative retrieval (GR) and recommendation decoding failures, constrained and filtered top-K, and normalization in constrained decoding.

**Verification key.**
- **[V]**: abstract or HTML read on arXiv or ACL this session. Numbers come from the paper as read.
- **[V-meta]**: bibliographic record verified (ACL Anthology, PMLR, JAIR or JACM). The mechanism is written from domain knowledge, not re-read this session.
- **[U]**: not verified this session. Treat as a pointer only.

**Closeness to RapidTinyRec certified exact top-K (CETK):**
- **direct**: same certificate principle, or exact top-K from a prefix-factored model using prefix bounds.
- **adjacent**: related mechanism (bounds, A*, certificates, beam-failure analysis) in another setting, or without exactness.
- **none**: background or context only.

Our method, for reference: s(i) = s(a) + s(b|a) + s(c|ab), with every term ≤ 0. The prefix bound is U(a,b) = s(a) + s(b|a).
- **Forward 1** is a tree-attention pass over the prompt plus all 88 first codes. It yields U for all ~2.5k prefixes.
- **Forward 2** expands the top-M prefixes by U.
- **Certificate**: the K-th exact score is ≥ the maximum U over unexpanded multi-child prefixes.
- **Trie normalization**: a single-child prefix has s(c|ab) = 0, so its item is already exact.

---

## 0. Bottom line

1. **The certificate principle is not new.** It is folklore, and one 2026 GR paper states our exact certificate for SID recommenders:
   - **NMT exact search.** Stahlberg & Byrne 2019 use DFS pruning with "log-probs ≤ 0". Stahlberg, Kulikov & Kumar 2022 give an exact n-best DFS whose bound is the n-th best complete score.
   - **Optimal stopping and best-first beam search.** Huang et al. 2017; Meister et al. 2020.
   - **Uniform-cost search over probabilistic label trees.** Jasinska-Kobus et al. 2020.
   - **Bound-based safe top-k in IR and databases.** Fagin's Threshold Algorithm (TA); WAND.
   - **A\* with non-positive global scores.** Lee, Lewis & Zettlemoyer 2016.
   - **The SID prior.** "Beyond the Beam: Constructive Repair and Candidate Completion for Generative Recommendation" (Zhao et al., arXiv 2609.33745, 27 Sep 2026). Its "global Top-K certificate" is t_K > max over unevaluated items of U_i, with U_i = (log-prob of the item's longest evaluated prefix) + d_i, "because the unevaluated suffix contributes only nonpositive terms". It cites Huang et al. 2017 and Meister et al. 2020 for this.

   **This is the closest prior and must be cited.**

2. **What remains defensibly new for us.** I found no paper with any of the following:
   - (a) A **primary decoder** for an LLM-scale SID recommender that **always** returns the certified global top-K under the model. Beyond the Beam bolts completion onto a T5/LC-Rec beam, with a capped budget and partial certification: Top-20 certified for 4.5–77.6% of test queries at budget 80.
   - (b) A **complete level-2 frontier**: every (a,b) bound comes from one tree-attention forward, not from whatever prefixes a beam happened to visit.
   - (c) Prefix-level expansion: every child scored in one batched forward, instead of teacher-forcing whole items in groups of 20.
   - (d) **Trie normalization as a cost lever.** Single-child prefixes need no expansion, and the same normalization gives two-sided bounds (§5, Prop. 3).
   - (e) An expansion-count analysis showing ≈K expansions and linear cost in K.
   - (f) Measured beam search error vs. exact search at K up to 500.
   - (g) The normalization choice moving HR@10 by more than 1 point. Exactness lets you attribute the change to the *model*, not to search.
3. **Beam-loss literature for SID and GR is now dense (2026).**
   - First-step coverage: 2611.01072.
   - Beam-aware training: BEAR (SIGIR'26) and APAO (KDD'26).
   - Item-supported decoding: ISD (2607.24995).
   - Finite-beam reachability: 2610.12977.
   - Long-tail loss: CAPE (2610.29298).
   - Brute-force exhaustive reference: DiGR (2602.07847), 5.05 s per user.
   - Prefix-retention optimization: PRO (2606.09241).
   - Lookahead guidance: PAG (SIGIR'24). It found that beam 1000 still "cannot meet the brute force decoding performance."

   All of them treat beam loss with training changes, re-allocated width or heuristics. **None besides Beyond the Beam certifies.** This literature is our motivation section.

---

## 1. Direct priors

### 1.1 Beyond the Beam: Constructive Repair and Candidate Completion for Generative Recommendation **[V]**
**Bibliographic.**
- arXiv 2609.33745 v1, 27 Sep 2026.
- Authors: Zijun Zhao, Peng Zhang, Gang Zhang, Yuanchi Ma, Hui He, Zhendong Niu (BIT, Tsinghua, SMU).

**Mechanism.**
- The setting is catalog expansion: new items get semantic IDs (SIDs) while old ones stay fixed. Repair is posed as a min-cost-flow problem.
- At inference, the score is F(i|h) = complete-SID log-likelihood + d_i, an item-level collaborative/new-item correction known before decoding.
- **Bound-guided completion.** For each item, take the cumulative log-prob u_i of its *longest evaluated prefix*, and set U_i = u_i + d_i.
- E_0 is "every complete catalog leaf scored in the terminal beam expansion, including leaves outside the returned beam."
- Completion then teacher-forces whole items in batches of 20, choosing the batch with the largest total U, with at most 80 extra items per query.
- **Certificate (App. B.6):** if the K-th largest exact score among evaluated items is ≥ every unevaluated bound, the returned list is the global Top-K under F. They also prove gap monotonicity: Δ(E) is non-increasing as E grows.
- For ties they use a numeric tolerance of 1e-3 and state that it is "not a formal floating-point error bound."

**Results.**
- Backbones: T5 with 4-token IDs and beam 40, plus LC-Rec. Datasets: Amazon Beauty, Tools, Toys.
- Top-20 certification rates (test, bound priority): 77.64% Beauty, 4.45% Tools, 61.53% Toys. Collaborative priority gives 45.40%, 0.71% and 31.95%.
- Beauty validation, as the budget grows from 20 to 160 items: 30.2%, 41.2%, 56.8%, 73.0%, 98.4%.
- Bound priority cuts extra evaluations on Beauty from 60.69 to 51.08.
- Latency is about 10–11 ms per query batched on an RTX 3090.
- They cite Huang et al. 2017 and Meister et al. 2020 for the stopping rule. They do not cite Stahlberg, A\*, PLT or Fagin.

**Closeness: DIRECT, the closest prior.** Same certificate. The differences:
1. Purpose: post-beam candidate completion for new items vs. our primary exact decoder.
2. Bound source: beam-visited prefixes only. Items whose first code was pruned have only a level-1 or a weak bound. Ours is a complete level-2 frontier from one forward.
3. Unit of expansion: whole items via teacher forcing vs. prefixes, all children at once, via tree attention.
4. Certification is partial and budget-capped; ours is designed for 100% certification.
5. They use a combined score with d_i; ours is the pure model score with several normalizations.
6. Small T5 vs. a 1.5B LLM.
7. They have no normalization-driven cost analysis and no large-K study.

### 1.2 Stahlberg, Kulikov & Kumar, "Uncertainty Determines the Adequacy of the Mode and the Tractability of Decoding in Sequence-to-Sequence Models" **[V]**
**Bibliographic.** ACL 2022, arXiv 2204.00471.

**Mechanism.**
- Exact **n-best** DFS: the n best complete hypotheses are kept in a priority queue, and the n-th best score is the pruning bound γ ("if p′ > γ then" recurse).
- Exactness rests on monotone, non-positive log-prob increments.
- Cost is counted in explored states, capped at 1M.

**Result.**
- n=1 always finishes.
- n=100: about 90% of MT-deen sentences finish under the cap; GEC finishes almost always, under 10k states.
- Explored states correlate with intrinsic uncertainty (Spearman ~0.18–0.20).
- Exact and beam 100-best mass differ by less than 0.001 in most setups.

**Closeness: DIRECT (generic seq2seq).** This is exact top-n under the same bound. The differences:
- Unbounded-length outputs over the full vocabulary, so DFS is exponential. Ours is fixed depth 3 over a trie, so it is batch-friendly.
- They have no batched or frontier expansion and no explicit "certificate gap" reporting.

Their tractability-vs-entropy finding maps onto our "expansions ≈ density of scores near τ_K" (§5).

### 1.3 Stahlberg & Byrne, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?" **[V]**
**Bibliographic.** EMNLP 2019, arXiv 1908.10090.

**Mechanism.**
- Exact top-1 search mixes beam search and DFS. The bound γ is initialized to the beam-search score and raised whenever a better complete hypothesis appears.
- "partial hypotheses can be safely discarded once their score drops below the log-probability of any complete hypothesis."
- EOS is tried first so γ rises early.

**Result.**
- Transformer on WMT15 En-De: beam (even at 100) usually misses the global best.
- For more than 50% of sentences the global best is the empty translation.
- Some length-constrained exact runs exceeded a day per sentence.

**Closeness: DIRECT (top-1).** Two ideas to borrow: an incumbent from beam as the initial bound, and "search error vs. model error" as the framing.

### 1.4 Jasinska-Kobus, Wydmuch, Dembczyński, Kuznetsov & Busa-Fekete, "Probabilistic Label Trees for Extreme Multi-label Classification" **[V]**
**Bibliographic.** arXiv 2009.11218, Sep 2020. PLTs were introduced in Jasinska et al., ICML 2016, PMLR 48 [V-meta].

**Mechanism.**
- Top-k prediction uses "a variant of uniform-cost search": a priority queue keyed on the path-product estimate, where each child's estimate is the parent's times the node classifier output.
- "Clearly, the priority queue guarantees" the k labels with the highest estimated marginals.
- Correctness relies on node estimates in [0,1], so path products are non-increasing.

**Result.** Exact top-k under the tree model.

**Closeness: DIRECT (analog in extreme multi-label classification, XMC).** Same best-first-with-multiplicative-≤1-bound idea. The differences:
- Node classifiers are cheap, independent per-node models.
- There is no LLM, no batching and no certificate reporting.

### 1.5 Huang, Zhao & Ma, "When to Finish? Optimal Beam Search for Neural Text Generation (modulo beam size)" **[V]**
**Bibliographic.** EMNLP 2017, arXiv 1809.00069.

**Mechanism.** Stop beam search as soon as the best finished hypothesis provably beats every unfinished one; monotone scores make this a certificate "modulo beam size". A bounded length reward keeps the stopping rule optimal.

**Result.** Never finishes later than the baseline; better BLEU.

**Closeness: ADJACENT.** The certificate holds only within the beam's search space. Beyond the Beam cites it.

### 1.6 Meister, Vieira & Cotterell, "Best-First Beam Search" **[V]**
**Bibliographic.** TACL 2020, arXiv 2007.03909.

**Mechanism.**
- Uses a priority-queue agenda. With a monotone scoring function, it returns the same set as beam search and can stop early.
- Gives monotone approximations for length normalization and MMI.
- Its framework treats beam search, best-first search and A\* as one family.

**Result.** Up to 10× faster than standard beam search.

**Closeness: ADJACENT.** It is a best-first agenda but beam-equivalent, not exact. Cite it for the monotonicity requirement, which length normalization breaks; that matters for our normalization choices.

### 1.7 Lee, Lewis & Zettlemoyer, "Global Neural CCG Parsing with Optimality Guarantees" **[V]**
**Bibliographic.** EMNLP 2016, D16-1262, arXiv 1607.01432.

**Mechanism.**
- A\* over subtrees. The global tree-LSTM score is a log-sigmoid, "ensures that it is non-positive—an important property for inference."
- Upper bounds from the locally factored model therefore remain admissible for the full model.

**Result.**
- Optimal parse for 99.9% of held-out sentences.
- 190 subtrees explored on average, 27.1 sentences per second.

**Closeness: DIRECT-ANALOG.**
- It is the clearest prior statement that *non-positive* model terms make a cheap factored bound admissible, giving an optimality guarantee for a neural model.
- It is also the precedent for engineering a model term to be ≤ 0 so that it can be certified. Our trie normalization makes the level-3 term exactly 0 on single-child prefixes, which is the same spirit.

---

## 2. Exact and optimal decoding in neural text generation (background and adjacent)

| # | Paper | Venue / ID / date | Mechanism | Result | Closeness |
|---|---|---|---|---|---|
| 2.1 | Forster, Meister & Cotterell, "Searching for Search Errors in Neural Morphological Inflection" **[V]** | EACL 2021, arXiv 2102.08424 | Exact inference (Dijkstra in SGNMT per secondary snippet; abstract says "exact inference"). Monotone decreasing scores mean the first EOS popped is optimal. Pruned with an incumbent (empty string or beam output). | Empty string is "almost never" the mode for inflection; greedy often finds the global optimum. | adjacent |
| 2.2 | Meister, Vieira & Cotterell, "If beam search is the answer, what was the question?" **[V]** | EMNLP 2020, arXiv 2010.02650 | Beam search is the exact solution of a UID-regularized objective. | Exact MAP gives poor text; UID adherence correlates with BLEU. | none (context: exact ≠ better in NLG; in recsys exact helps, per our HR results and DiGR's exhaustive NDCG gain) |
| 2.3 | Yang, Huang & Ma, "Breaking the Beam Search Curse" **[V]** | EMNLP 2018, arXiv 1808.09582 | Rescoring plus optimal stopping criteria. | +2.0 BLEU over length normalization. | adjacent |
| 2.4 | Murray & Chiang, "Correcting Length Bias in NMT" **[V]** | WMT 2018, arXiv 1808.10006 | Length bias and the beam problem both come from label bias; per-word reward fixes both. | Brevity fix nearly removes the beam problem. | adjacent (normalization) |
| 2.5 | He, Sun, Jia & Li, "Empirical Analysis of Beam Search Curse and Search Errors with Model Errors in NMT" **[V]** | EAMT 2023 | Exact-search analysis. | Fewer than 30% of sentences show the curse or search errors; better search is unlikely to help MT. | none (contrast: our domain shows search error grows with K) |
| 2.6 | Chen, Gilroy, Maletti, May & Knight, "Recurrent Neural Networks as Weighted Language Recognizers" **[V]** | NAACL 2018, arXiv 1711.05408 | Complexity. | Highest-weighted string: undecidable in general; for consistent RNNs decidable, but solution length can surpass all computable bounds; NP-complete and APX-hard when restricted to polynomial length. | theory (contrast: fixed-depth trie makes exact top-K polynomial, O(#prefixes) per query) |
| 2.7 | Lu et al., "NeuroLogic A\*esque Decoding" **[V]** | arXiv 2112.08726 (NAACL 2022 [U]) | A\*-like lookahead future-cost heuristics (rollouts) for constraint satisfaction. | SOTA on five constrained generation tasks. | adjacent (heuristic is **not** admissible; no exactness) |
| 2.8 | "A\*-Decoding: Token-Efficient Inference Scaling" **[V]** | arXiv 2505.13672, May 2025 | A\* over reasoning steps with a PRM heuristic; explicitly trades admissibility for informativeness. | Matches best-of-N and particle filtering with up to 3× fewer tokens. | none/adjacent |
| 2.9 | Dobrowolski et al., "A\* Decoding for MT in LLMs (SRPOL WMT25)" **[V]** | WMT 2025, pp. 666–670 | A\* candidate generation from EuroLLM plus QE reranking. | Better MT. | none |
| 2.10 | Grubisic, Cummins, Seeker & Leather, "Priority Sampling of LLMs for Compilers" **[V]** | arXiv 2402.18734, Feb 2024 | Deterministic best-first enumeration: expand the highest-probability unexpanded token in an augmented tree. Gives unique samples in order of confidence; regex constraints supported. | Beats nucleus sampling at any number of samples. | adjacent (best-first enumeration of top sequences, no certificate) |
| 2.11 | Kool, van Hoof & Welling, "Stochastic Beams and Where to Find Them: Gumbel-Top-k" **[V]** | ICML 2019, arXiv 1903.06059 | Gumbel-top-k on a factorized sequence model gives exact samples without replacement (stochastic beam search). | Linear in k and length; low-variance estimators. | adjacent (perturbed scores + top-k = exact *sampled* top-k; our certificate would carry over to Gumbel-perturbed scores if perturbations are bounded per prefix via the Gumbel-max-of-subtree trick) |
| 2.12 | Shi, Bieber & Sutton, "Incremental Sampling Without Replacement for Sequence Models" **[V]** | arXiv 2002.09067, 2020 (ICML 2020 [U]) | Incremental sampling without replacement over exponentially large output spaces (trie-based, per the paper's UniqueRandomizer [U]). | Program synthesis and combinatorial optimization. | adjacent |
| 2.13 | Meister, Forster & Cotterell, "Determinantal Beam Search" **[V]** | ACL-IJCNLP 2021, arXiv 2106.07400 | Each beam step is a subdeterminant maximization (DPP) for diverse sets. | Competitive diverse decoding. | adjacent (diversity-constrained top-K; not exact) |
| 2.14 | Andor et al., "Globally Normalized Transition-Based Neural Networks" **[V]** | ACL 2016, arXiv 1603.06042 | Label bias: "globally normalized models can be strictly more expressive than locally normalized." | SOTA tagging and parsing. | adjacent (normalization theory behind trie-renormalized vs full-vocab scoring) |

---

## 3. A\*, k-best and certificates (parsing, MT, combinatorial search)

| # | Paper | Venue / date | Mechanism | Result | Closeness |
|---|---|---|---|---|---|
| 3.1 | Klein & Manning, "A\* Parsing: Fast Exact Viterbi Parse Selection" **[V-meta]** | NAACL 2003, pp. 119–126 | A\* with admissible, monotone outside estimates (grammar projections, context summaries) returns the Viterbi parse. | Large reductions in edges processed (exact figure not re-read). | adjacent (template: admissible outside estimate = our U(p) with outside = 0) |
| 3.2 | Pauls & Klein, "K-Best A\* Parsing" **[V-meta]** | ACL-IJCNLP 2009, pp. 958–966 | k-best A\*: pop k complete derivations from an agenda ordered by inside + admissible outside. | Exact k-best with far fewer items than exhaustive search. | adjacent→direct template for certified k-best |
| 3.3 | Huang & Chiang, "Better k-best Parsing" **[V-meta]** | IWPT 2005, pp. 53–64 | Lazy k-best extraction from hypergraphs. | Exact k-best in O(E + D·k log k) [from knowledge]. | adjacent (exact *ordering* once scores exist) |
| 3.4 | Lewis & Steedman, "A\* CCG Parsing with a Supertag-factored Model" **[V-meta]** | EMNLP 2014, pp. 990–1000 | Supertag-factored model; agenda ordered by inside + "heuristic upper-bound on the outside probability" (secondary snippet). | Fast exact parsing. | adjacent (factored-model upper bound) |
| 3.5 | Mohri & Riley, "An Efficient Algorithm for the N-Best-Strings Problem" **[V-meta]** | ICSLP/Interspeech 2002 | N-best *distinct* strings in a weighted automaton via on-demand determinization. | First algorithm to remove redundancy before n-best extraction. | adjacent (exact n-best over a finite automaton; our trie is a finite acyclic automaton) |
| 3.6 | Rush, Chang & Collins, "Optimal Beam Search for Machine Translation" **[V-meta + abstract via search]** | EMNLP 2013, pp. 210–221 | Beam search provides a lower bound and Lagrangian relaxation an upper bound; matching bounds give a certificate. | Exact solutions with optimality certificates on 99% of examples; 3.5× (phrase-based) and 4× (syntax-based) faster than an optimized constraint-based exact decoder. | adjacent (beam + bound certificate; our "beam as incumbent + prefix bounds" is the neural/trie analog) |
| 3.7 | Rush & Collins, "A Tutorial on Dual Decomposition and Lagrangian Relaxation for Inference in NLP" **[V]** | JAIR 45:305–362, 2012; arXiv 1405.5208 | Lagrangian relaxation and dual decomposition; certificate when the relaxed solution satisfies the relaxed constraints (zero duality gap). | Survey with guarantees. | adjacent (route to *constrained* certified top-K, §4) |
| 3.8 | Rush, Sontag, Collins & Jaakkola, "On Dual Decomposition and LP Relaxations for NLP" **[V via search snippets]** | EMNLP 2010 | Dual decomposition solves an LP relaxation; "certificate of optimality when they recover the exact solution." | High exactness rates [figures not re-read]. | adjacent |
| 3.9 | Koo, Rush, Collins, Jaakkola & Sontag, "Dual Decomposition for Parsing with Non-Projective Head Automata" **[U, snippet only]** | EMNLP 2010 | Same idea. | "very often give an exact solution … certificate of optimality." | adjacent |
| 3.10 | Dechter & Pearl, "Generalized Best-First Search Strategies and the Optimality of A\*" **[V-meta]** | JACM 32(3):505–536, 1985 | Optimal-efficiency results: with a consistent heuristic, any admissible algorithm must expand every node with f < C\*; A\* expands no more (up to ties). | Classic. | theory (backs our Prop. 2) |
| 3.11 | Hansen & Zhou, "Anytime Heuristic Search" **[V-meta]** | JAIR 28:267–297, 2007 | Weighted A\* turned anytime: quick solution, then continue to improve the solution and the sub-optimality bound. | Classic. | adjacent (ε-certificate when the budget runs out) |
| 3.12 | Agostinelli et al., DeepCubeA, "Solving the Rubik's Cube with deep RL and search" **[V via secondary sources]** | Nature Machine Intelligence 2019 [journal per secondary] | Batch Weighted A\* (BWAS): pop N nodes per iteration and evaluate all successor heuristics in one batched GPU inference. | Practical neural-guided A\*. | adjacent (our two-round batched best-first is BWAS with λ=1 and a data-dependent batch M) |
| 3.13 | "Learning Empirically Admissible Neural Heuristics for Combinatorial Search" **[V, HF page; authors not shown]** | arXiv 2606.04860, Jun 2026 | Underestimating Bellman targets plus an asymmetric pinball loss plus a calibration offset δ (max validation overestimate). | 0 observed violations; admissibility only empirical (Thm 2 needs a sup bound). | adjacent (learned tighter-than-zero bounds give a *probabilistic* certificate) |

---

## 4. Bound-based safe top-k outside NLP (IR, databases, MIPS, XMC, tree retrieval)

| # | Paper | Venue / date | Mechanism | Result | Closeness |
|---|---|---|---|---|---|
| 4.1 | Fagin, Lotem & Naor, "Optimal Aggregation Algorithms for Middleware" (Threshold Algorithm) **[V via search]** | PODS 2001; JCSS 66 (2003) 614–656; arXiv cs/0204046 | Sorted access per list. Stop when the k-th seen aggregate ≥ the threshold (aggregate of the last-seen values, an upper bound on unseen objects). | **Instance optimal** for every monotone aggregation function and every database. | DIRECT-ANALOG (same stopping certificate; the canonical citation for instance optimality of bound-based top-k) |
| 4.2 | Broder et al., "Efficient Query Evaluation Using a Two-Level Retrieval Process" (WAND) **[V via search]** | CIKM 2003 | Per-term score upper bounds; fully evaluate a doc only if its upper bound clears the current top-k threshold. "Safe" top-k. | Over 90% fewer full evaluations on TREC with almost no loss. | adjacent (two-level: cheap bound pass, then exact scoring of survivors, which mirrors our forward 1 / forward 2) |
| 4.3 | Ram & Gray, "Maximum Inner-Product Search using Cone Trees" **[V via search]** | KDD 2012; arXiv 1202.6101 | Branch-and-bound exact MIPS with ball and cone bounds. | Up to 5 orders of magnitude faster than linear scan. | adjacent (exact top-k via node bounds) |
| 4.4 | Liu, Yu & Lengerich, "CSV-Decode: Certifiable Sub-Vocabulary Decoding" **[V]** | arXiv 2511.21702, Nov 2025 | Cluster the vocabulary embeddings; certify per-step top-k tokens when every unopened cluster's centroid+radius logit bound < the k-th opened logit; fall back by opening clusters. | 2.7–5× speedups; certification rate 96.4–98.2% (inconsistent across sections). | adjacent (same certificate shape at token level; geometric logit bounds) |
| 4.5 | Zhu et al., "Learning Tree-based Deep Model for Recommender Systems" (TDM) **[V]** | KDD 2018, arXiv 1801.02294 | Item tree; coarse-to-fine beam over user-node scores. Max-heap-like node semantics (per Zhuo et al.). | Taobao online gains. | adjacent |
| 4.6 | Zhuo et al., "Learning Optimal Tree Models Under Beam Search" **[V]** | ICML 2020, arXiv 2006.15408 | Defines top-k **Bayes optimality under beam search**. Prop. 1: if node scores equal p\*(z_n\|x), the max over the node's leaves, beam retrieves the exact top-k. Training is made beam-aware. | Beats TDM, JTM and PLT. | adjacent→direct theory: under trie normalization our U(p) is *exactly* the max-heap value for single-child prefixes and within log n_p for multi-child ones (§5 Prop. 3). This explains why beam error shrinks under trie normalization. |

---

## 5. Generative retrieval and SID recommendation: decoding failures and guarantees

| # | Paper | Venue / ID / date | Mechanism | Result | Closeness |
|---|---|---|---|---|---|
| 5.1 | Yao & Zhu, "Where Beam Search Loses Top-Ranked Items: First-Step Coverage in Generative Recommendation" **[V]** | arXiv 2611.01072, 31 Oct 2026 (Ant Group) | Exhaustive offline scoring of every legal SID gives the exact top-K reference. Every pruned exact top-10 path is first lost at the **first** step. Fix: front-loaded widths, e.g. Front-151 = (100, 30, 20, 10). | TIGER/GRID on Beauty, Sports, Toys × RQ-KM/RQ-VAE. Search Recall@10, Beam-50 → Front-151: 0.893→0.981 (Beauty RQ-KM), 0.884→0.993 (Sports RQ-VAE). Front-151 reproduces Beam-100's ordered top-10 for all 77,373 users; only ~6.4% latency gain (forward-dominated). No bounds or certificates. | DIRECT motivation. Our forward 1 expands **all** first codes, so level-1 search error is zero by construction, which is their bottleneck. |
| 5.2 | Zhao et al., DiGR "From Token Generation to Item Ranking: Direct Generative Recommendation with Semantic IDs" (v1 title "SimGR: Escaping the Pitfalls of Generative Decoding in LLM-based Recommendation") **[V]** | arXiv 2602.07847 v1 8 Feb 2026, v2 16 Aug 2026 | Brute-force exhaustive AR scoring of LC-Rec (Qwen3-4B) as the reference. Thm 1: per-position marginals are not ranking-sufficient when \|I\| > 1 + Σ(\|V_l\| − 1). v1 Thm 4.1 bounds expected top-K overlap by P(all tokens of an item rank ≤ B). | Instruments: exhaustive NDCG@10 0.0842, 5.052 s/user; beam 50 is 0.0802 (90.28% overlap, 0.683 s); beam 100 is 0.0816 (94.76%, 0.981 s). v1: union-bound estimate of about 66% joint survival at L=4, B=20. No bounds or best-first. | DIRECT motivation and the baseline to beat: our exact top-K should be ≪ 5 s and near beam latency. |
| 5.3 | Yang et al., BEAR "Beam-Search-Aware Optimization for Recommendation with LLMs" **[V]** | SIGIR 2026, arXiv 2601.22925 | Training regularizer: each positive item's token ranks within top-B at every step (necessary condition for beam survival). | v1: "over 80% of positive items with top-B overall probability are pruned before reaching the final recommendations" (88.28% Book, 84.39% Toy, B=10; Llama-3.2-3B/BIGRec). | adjacent (training-side fix for search error; exact search removes the need) |
| 5.4 | Yu et al., APAO "Adaptive Prefix-Aware Optimization" **[V]** | KDD 2026, arXiv 2603.02730 | Prefix-level losses plus adaptive worst-prefix training so the correct item survives beam pruning. | Gains across GR backbones. | adjacent |
| 5.5 | Wang, He, Li & Sundaram, ISD "Understanding Semantic IDs: From Item Representation to Item Selection" **[V]** | arXiv 2607.24995, Jul 2026 | Item-Supported Decoding: a user-specific item ranking supports matching SID prefixes before beam pruning and orders the final items. | TIGER keeps only 29.9% of held-out targets that were plausible pre-filtering; NDCG@10 up to +31.2%. | adjacent (heuristic lookahead, no guarantee) |
| 5.6 | Sun et al., "Beyond Semantic IDs: Finite-Beam Reachability and Continuous-Query Generation" **[V]** | arXiv 2610.12977, Oct 2026 | Empirical "finite-beam reachability ceiling" (TargetSupport@B). Proposes continuous-query generation with exact dense scoring. | ML-1M TIGER TargetInOutput@100 66.7%, 76.3% at 200. No formal bound. Notes expanding the internal beam when filtering leaves too few items. | adjacent |
| 5.7 | Huang et al., CAPE "Generative Recommenders Forget the Long Tail" **[V]** | arXiv 2610.29298, Oct 2026 | Beam keeps prefixes by subtree mass, so tail items are pruned early; a co-occurrence channel adds out-of-beam tail items. | Better long-tail accuracy. | adjacent (beam bias toward head; exact top-K removes the *search* part of this bias) |
| 5.8 | Chen et al., PRO "Closing the Indexing-Decoding Gap in Multimodal GR via Prefix Retention Optimization" **[V]** | arXiv 2606.09241, Jun 2026 | Prefix ranking distillation, larger deeper codebooks, and geometric score fusion so targets survive the beam. | Gains on 9 multimodal tasks. | adjacent |
| 5.9 | Zeng, Luo & Zamani, PAG "Planning Ahead in Generative Retrieval" **[V]** | SIGIR 2024, arXiv 2404.14600 | Set-based (bag-of-words) simultaneous decoding gives doc scores. Each prefix's beam score = sequential prefix score + **max** set-based score among top-n docs under the prefix, a lookahead heuristic (not admissible). | "Even with a beam size of 1000, RIPOR with constrained beam search cannot meet the brute force decoding performance." PAG: MRR@10 0.385 vs 0.333; beam 100; 22× latency speedup. | adjacent (closest GR "A\*-heuristic" analog; max-over-subtree as h) |
| 5.10 | Zeng et al., RIPOR "Scalable and Effective Generative Information Retrieval" **[V]** | WWW 2024 (per repo), arXiv 2311.09134 | Prefix-oriented ranking optimization (every prefix of a relevant docid must stay in the top-k) plus relevance-based docids. | +30.5% MRR@10 on MS MARCO; ablation without prefix optimization 0.333→0.280. | adjacent |
| 5.11 | Wu et al., "Constrained Auto-Regressive Decoding Constrains Generative Retrieval" **[V]** | SIGIR 2025, arXiv 2504.09935 | Bayes-optimal GR model plus corpus constraints. Error lower bound = KL between ground-truth and model step-wise marginals (tied to the Simpson diversity index). Beam with marginal (subtree-sum) values is sub-optimal: Thm C.1 says top-λk recall < 0.5 + max{0.65 − 0.15/λ, 0} while top-1 precision = 1. Constrained step "zeros out invalid tokens and then re-normalizes". | MS MARCO first-step relevant-branch R@50 is 53.7–67.5. | DIRECT theory context: their beam failure arises from *sum* (marginal) prefix values. Our bound uses *max*-type U(p) ≥ best descendant, which is what makes exact best-first correct. |
| 5.12 | Wu et al., "Generative Retrieval as Multi-Vector Dense Retrieval" **[V]** | SIGIR 2024, arXiv 2404.00684 | GR scoring = sum of query-doc token products weighted by alignment, so GR is a special case of multi-vector dense retrieval. | Theory. | adjacent (supports viewing exact GR top-K as MIPS-like exact retrieval) |
| 5.13 | Zhang et al., TSGen "Generative Retrieval via Term Set Generation" **[V]** | arXiv 2305.13859 (v3 Apr 2024) | Permutation-invariant term-set docids to avoid "false pruning". | Gains on benchmarks. | adjacent (changes the identifier; we keep it and fix the search) |
| 5.14 | Li et al., LTRGR "Learning to Rank in Generative Retrieval" **[V]** | AAAI 2024, arXiv 2306.15222 | Rank loss in a second training phase; no inference cost. | SOTA GR on 3 benchmarks. | none |
| 5.15 | De Cao et al., GENRE "Autoregressive Entity Retrieval" **[V]** | ICLR 2021, arXiv 2010.00904 | AR entity-name generation with trie-constrained beam search (trie detail from knowledge). | 20+ datasets. | none (origin of trie-constrained beam) |
| 5.16 | Randrianarivo, Renaud & Allal, "Disentangling Paradigm, Identifier, and Decoding in GR" **[V]** | arXiv 2610.08716, Oct 2026 | One-pass scoring of every document by its code probabilities (masked-diffusion GR). | Decoding alone moves diffusion Hit@1 by 6.6–13.7; AR "lead comes from the model, not beam search". | adjacent (exhaustive one-pass scoring vs search) |
| 5.17 | Liu et al., BONSAI "Beyond Fixed Depths and Widths: Optimizing Textual Decoding Tries in LLM-based GR" **[V]** | arXiv 2607.16633, Jul 2026 | Variable-length IDs; constrained branching at shallow levels "drastically improve[s] the success rate of constrained beam search". | Up to +21.6%. | adjacent (trie shape determines beam error *and* our certification cost via n_p) |
| 5.18 | Su et al., STATIC "Vectorizing the Trie" **[V]** | KDD 2026, arXiv 2602.22647 | CSR transition-matrix trie for constrained decoding to business-defined subsets ("fresh content or a product category"). | +0.033 ms/step; 948× vs CPU trie; deployed at YouTube scale. | adjacent (filtered retrieval; full-vocab log-softmax then mask, per ledger) |
| 5.19 | xGR (JD) early sorting termination; GR4AD (Kuaishou) TopK pre-cut **[V in ../lit_ledger.md]** | arXiv 2512.11529; 2602.22732 | Exact per-step *selection* shortcuts (min-heap early termination, per-beam then global top-k). | Large QPS gains. | adjacent (exact selection within beam, not exact search) |
| 5.20 | Rajput et al., TIGER "Recommender Systems with Generative Retrieval" **[V-meta]** | NeurIPS 2023, arXiv 2305.05065 | RQ-VAE semantic IDs plus beam search. | Origin of the SID-GR setting. | none |

---

## 6. Constrained and filtered top-K, and normalization / calibration in constrained decoding

| # | Paper | Venue / date | Mechanism | Result | Closeness |
|---|---|---|---|---|---|
| 6.1 | Park et al., "Grammar-Aligned Decoding" (ASAp) **[V]** | NeurIPS 2024, arXiv 2405.21047 | Grammar-constrained decoding (GCD) masking distorts the LM distribution. ASAp over-approximates expected future grammaticality from prior samples and converges to the constrained conditional. | Higher likelihood outputs, still grammatical. | adjacent (formalizes that *local* renormalization ≠ conditioning on the constraint, which is our trie- vs full-vocab-normalization issue) |
| 6.2 | Özgü, Wu, Van den Broeck & Kim, "The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding" (SHIM) **[V]** | arXiv 2608.10137, Aug 2026 (v2 Oct) | An offline-trained next-token correction conditioned on parser state reduces GCD bias. | Closer to the grammar-conditioned distribution at near-masking speed. | adjacent |
| 6.3 | Bao et al., D3 "Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation" **[V]** | EMNLP 2024, arXiv 2406.14900 | "Ghost tokens" with p≈1 inflate length-normalized item scores. D3 removes length normalization on them and adds a diversity assistant. | Accuracy and diversity gains. | DIRECT for our normalization finding. Trie normalization makes forced tokens exactly 0, the principled version of D3's fix, and also what makes single-child prefixes free in CETK. |
| 6.4 | Zhu et al., "Learning Better Reasoning for GR with Semantic IDs" **[V]** | arXiv 2609.29973, Sep 2026 | Scores = sum of per-level log-probs "after masking and renormalizing over legal trie children". | No normalization ablation. | adjacent (evidence that trie renormalization is in use, unablated) |
| 6.5 | Netflix, "Personalized LLM Recommendation with SIDs: AR beam vs MTP" **[V in ledger]** | arXiv 2610.15864, Oct 2026 | AR constrained beam with *length-normalized* log-prob. | AR beats MTP on quality. | adjacent (yet another normalization choice; not monotone, so best-first/exactness needs care, per Meister 2020) |
| 6.6 | GRACE (Meta) bitmask and Bloom targeting constraints; xGR mask-before-softmax **[V in ledger]** | arXiv 2608.00938; 2512.11529 | Filtered GR in production. xGR adds the mask to logits *before* softmax (renormalized over valid); STATIC log-softmaxes over the full vocabulary *then* masks. | — | adjacent (two filter semantics, see §8.2) |
| 6.7 | Wu et al., SIGIR 2025 (5.11) | — | Constraint-induced KL lower bound. | — | theory for normalization under constraints |

---

## 7. Comparison: closest priors vs. RapidTinyRec CETK

| Property | Stahlberg 2019/22 | PLT uniform-cost search | Fagin TA | Beyond the Beam (2609.33745) | 2611.01072 / DiGR | **RapidTinyRec CETK** |
|---|---|---|---|---|---|---|
| Model | NMT/GEC transformer | per-node classifiers | lists of grades | T5 / LC-Rec SID | TIGER / LC-Rec 4B | Qwen2-1.5B MiniOneRec SID |
| Output | top-1 / top-n strings | top-k labels | top-k objects | top-K items (combined score) | exact reference (offline) | top-K items (model score) |
| Bound | prefix log-prob (≤ 0) | path product (≤ 1) | threshold of last-seen | longest *beam-visited* prefix + d_i | none (brute force) | **every** (a,b) prefix from one forward |
| Expansion unit | one token, DFS | one node | one sorted access | whole item, teacher-forced ×20 | all items | all children of top-M prefixes in one tree-attention forward |
| Certification | always (with cap) | always | always | 4.5–77.6% (test, budget 80) | n/a | designed for 100% (2 forwards, plus lower-bound schedule §8.1) |
| Normalization lever | — | — | — | full-vocab | — | trie norm: single-child prefixes free, two-sided bounds |
| Cost scaling | exponential DFS | O(k · depth) nodes | instance-optimal | budget-capped | O(\|I\|) | ≈K prefix expansions (§9 Prop. 3) |

---

## 8. Proposals: classic ideas that make CETK cheaper or broader

### 8.1 Two-sided bounds from trie normalization: always certify in exactly 2 forwards, with no fallback round (**highest value**)
**Mechanism.**
1. Under trie normalization, a prefix p = (a,b) with n_p children satisfies max_c s(c|ab) ≥ −log n_p, because the children's probabilities sum to 1.
2. So the best item under p scores in [L(p), U(p)], with L(p) = U(p) − log n_p. For single-child prefixes L = U, which is exact.
3. After forward 1, set τ_L = the K-th largest L(p) over distinct prefixes. Each prefix contributes a distinct item, so τ_L ≤ τ\*, the true K-th score.
4. **Prune permanently** every prefix with U(p) < τ_L.
5. **Expand exactly S = {multi-child p : U(p) ≥ τ_L}** in forward 2. Every prefix the certificate could need is in S, so the certificate is guaranteed after forward 2.

**What it replaces.** The current fixed-M heuristic plus a possible third forward.

**Prior art.** This is a WAND/TA-style two-level scheme (Broder 2003; Fagin 2003). The use of a *normalization-derived lower bound* appears nowhere in the sources reviewed.

**Under full-vocab or codebook normalization** there is no lower bound: valid-child mass can be ≈0. M must be chosen adaptively, which is a concrete *cost* argument for trie normalization on top of its HR effect.

### 8.2 Filtered retrieval (availability masks, business rules)
The certificate stays sound under any leaf filter, because bounds come from ancestors only. Precompute per-prefix allowed-descendant counts (a bitset per request) and drop empty prefixes.

The cost depends on the **filter semantics**, which nobody in GR specifies:
- **(A) Score-then-filter.** Full or trie scores are unchanged (STATIC-like). U is valid, but L(p) is invalid unless the argmax child is allowed. Selective filters can therefore force more expansions, and beam must over-fetch. "Beyond Semantic IDs" notes that beams are expanded when filtering leaves too few items.
- **(B) Renormalize over the allowed sub-trie.** Forward 1 masks to allowed children (xGR-like mask before softmax). Two-sided bounds return with n_p = number of allowed children, and single-allowed-child prefixes become free. Certification cost stays ≈K for any filter.

Report both. CETK always returns exactly K valid items, whereas beam can return fewer than K or wrong items under selective filters. This is a measurable advantage, and I found no GR paper that measures beam degradation as a function of filter selectivity (open gap).

### 8.3 Diversity and fairness caps
- **Per-group caps** (≤ m items per first code a, per category, per seller) form a **partition matroid**, so greedy over items in non-increasing exact score is optimal.
- Best-first CETK emits items in certified order: pop an item once its score ≥ the max unexpanded U. The same certificate therefore gives the exact capped top-K.
- Once group a is full, all of a's prefixes are pruned, so caps *reduce* cost.
- Non-modular diversity (DPP, MMR; determinantal beam search, Meister 2021) is NP-hard. Options: Lagrangian relaxation of coverage constraints with a certificate when relaxed = primal (Rush & Collins 2012; Rush et al. 2013), or certified (1 − 1/e) greedy guarantees.

### 8.4 Large K
- Cost ≈ |{p : U(p) ≥ τ\*}|, linear in K plus the band term (§9).
- **Crossover:** once |S| approaches the number of multi-child prefixes, one exhaustive tree-attention forward over all multi-child prefixes is cheaper than bookkeeping. With ~2.5k prefixes and 1.4 children on average, a large share are single-child, so exhaustive is cheap. Report the crossover K.
- **Anytime ε-certificate** (Hansen & Zhou 2007). If the row budget runs out, return the top-K with gap ε = max unexpanded U − τ_K. Every missed item scores ≤ τ_K + ε, and certified overlap ≥ K − Σ_{p unexpanded, U(p) > τ_K} min(n_p, K). This gives a **certified recall lower bound** that beam search cannot report.
- **Incumbent warm start** (Stahlberg & Byrne 2019): use the beam-B result's K-th score as the initial τ. If beam was already exact, the certificate check costs nothing beyond forward 1.

### 8.5 Tighter-than-zero upper bounds (A\*-style heuristics)
- **(a) Already optimal for single-child prefixes** (trie normalization gives an exact 0). For multi-child prefixes the slack is ≤ log n_p, so there is little to gain. Better tries (BONSAI's shallow branching control; low fan-out at level 3) lower both beam error and CETK cost.
- **(b) Known additive item terms** (popularity priors, business boosts, Beyond the Beam's d_i) fold in as U(p) + max_{i∈p} d_i, which is exact and admissible.
- **(c) Learned bounds.** A small head on the level-2 node state from forward 1 predicts max_c s(c|ab) or the valid-mass log P(valid | ab) under full-vocab normalization.
  - Admissible only empirically ("likely-admissible": 2606.04860; conformal calibration).
  - Yields a **probabilistic certificate** (miscoverage ≤ δ) or, via a weighted-A\* style w·h, a bounded-suboptimal ε-certificate.
  - Useful only for full-vocab or codebook normalization, where 0 is loose because invalid level-3 mass is never subtracted.
- **(d) Geometric logit bounds** (CSV-Decode style centroid + radius on child LM-head rows). These need the level-3 hidden state, i.e. forward 2 itself, so they don't help the prefix bound. They could cut the LM-head cost inside forward 2 when the vocabulary is not already restricted.

### 8.6 Batched best-first as the formal frame
Our 2-round schedule is BWAS (DeepCubeA) / batched K-best A\* (Pauls & Klein 2009):
- the heuristic is h = 0 on level 3 (the uniform-cost search special case);
- the batch size M is set from bounds (§8.1) rather than fixed.

Cite Dechter & Pearl for optimal efficiency up to batching overshoot.

---

## 9. Theory we can cite or prove (statements plus proof sketches)

**Notation.**
- Prefix p = (a,b); U(p) = s(a) + s(b|a); items i ∈ p have s(i) = U(p) + s(c|ab) ≤ U(p).
- τ\* = the K-th largest s(i) over the catalog; n_p = number of children.
- Under trie normalization, single-child prefixes are exact.

**Prop 1 (soundness).** Let E be the set of exactly scored items and τ_E the K-th best in E. If τ_E ≥ max{U(p) : p multi-child, unexpanded}, then top-K(E) is a global top-K, up to ties.
- Proof: one line.
- Cite Stahlberg & Byrne 2019; Stahlberg et al. 2022; Huang et al. 2017; Fagin et al. 2003; Beyond the Beam Prop. 2.
- *State it as known.*

**Prop 2 (necessity; instance optimality of best-first).** Consider any algorithm that observes U(p) and n_p for unexpanded prefixes and certifies the top-K. It must expand every multi-child p with U(p) > τ\*.
- Adversary: with n_p ≥ 2, trie-normalized child probabilities can put mass 1 − ε on one child, giving an item scoring U(p) + log(1 − ε) > τ\*.
- Best-first by U expands exactly {p : U(p) ≥ τ\*} (plus ties), so it is optimal on every instance.
- Analogs: Dechter & Pearl 1985 (A\* optimal efficiency); Fagin et al. 2003 (TA instance optimality).
- *New in this setting, easy to prove.*

**Prop 3 (expansion count under trie normalization).**
- Let N\* = {multi-child p : U(p) > τ\*}. Each p ∈ N\* has a best item with s ≥ U(p) − log n_p > τ\* − log n_max, and distinct prefixes give distinct items.
- Hence |N\*| ≤ |{i : s(i) > τ\* − log n_max}| = (K − 1) + #{items with score in (τ\* − log n_max, τ\*]}.
- So expansions = K + (score density in a band of width log n_max below τ\*). This gives **linear-in-K cost**, matching our measured ≈K.
- The band term is the analog of Stahlberg et al. 2022's finding that exact search cost grows with uncertainty (flat score distributions).
- Corollary: the §8.1 schedule S satisfies |S| ≤ |{p : U(p) ≥ τ\* − log n_max}| when the top-K items lie in distinct prefixes.
- *Novel as far as found.*

**Prop 4 (beam search exactness condition).**
- Beam with level-2 width B (ranking prefixes by U) is exact for top-K iff every prefix holding a top-K item is among the top-B by U.
- **Sufficient:** B ≥ |{p : U(p) ≥ τ\*}|.
- At level 1, beam with width B₁ < 88 adds a first-step loss (2611.01072). CETK has zero level-1 loss because it expands all first codes.
- This is Zhuo et al. 2020's max-heap condition, made quantitative. Trie normalization makes U a max-heap–exact score on single-child prefixes and log n_p-approximate elsewhere. Full-vocab normalization lets U overestimate by −log(valid mass), which is unbounded, and so increases beam error.
- *This links our two empirical findings: normalization changes both search error and certification cost.*

**Prop 5 (certified overlap under budget).** If CETK stops with unexpanded set R, then |top-K(E) ∩ global top-K| ≥ K − Σ_{p∈R, U(p) > τ_E} min(n_p, K).
- Anytime guarantee; cf. Hansen & Zhou 2007.

**Prop 6 (complexity contrast).** General highest-probability string search for RNN LMs is undecidable, and NP-complete for polynomial-length strings (Chen et al. 2018). With a fixed-depth trie over |I| items, exact top-K costs at most #internal-prefix rows.
- This is trivial, but worth one sentence to pre-empt "exact decoding is intractable" reviewers.

**Cite for context.**
- Wu et al. SIGIR 2025 (marginal/sum prefix values make beam sub-optimal; constraint KL bound).
- DiGR Thm 1 (per-position marginals are not ranking-sufficient).
- Meister et al. 2020 (monotonicity needed for best-first; length normalization breaks it, so CETK is exact for sum-of-log-probs and trie-normalized scores, *not* for length-normalized scores without a monotone surrogate).
- Andor et al. 2016 and Murray & Chiang 2018 (label and length bias from local normalization).

---

## 10. Suggested positioning text (draft)

> Certifying an optimal output with monotone prefix bounds is classical. Examples: exact DFS and n-best search for NMT (Stahlberg & Byrne 2019; Stahlberg et al. 2022), optimal stopping in beam search (Huang et al. 2017; Meister et al. 2020), A\* with non-positive scores (Lee et al. 2016), uniform-cost search in label trees (Jasinska-Kobus et al. 2020), and the threshold algorithm (Fagin et al. 2003). Concurrently, Zhao et al. (2026) apply the same certificate to complete beam-search candidates for new items in SID recommenders, certifying 5–78% of queries under a fixed budget. We instead make certification the decoder itself. One tree-attention forward bounds every prefix of the SID trie, and a second expands only the prefixes the certificate requires. Under trie normalization this is provably about K prefixes and always succeeds in two forwards. As a result, every reported metric reflects the model's own top-K with zero search error, at latency comparable to beam search.

---

## 11. Verification log and open gaps
- **Not found despite targeted searches** (extended web search, 6 queries): any paper that (i) uses best-first or branch-and-bound as the *primary* SID or GR decoder for LLM recommenders; (ii) uses tree attention to score a complete level-2 frontier; (iii) measures beam degradation vs. filter selectivity; (iv) ablates trie- vs full-vocab normalization for ranking quality. Item (iv) is consistent with the earlier ledger (../lit_ledger.md §0).
- **[U] / partially verified:**
  - Exact certification rates in Rush et al. 2010 and Koo et al. 2010.
  - The Klein & Manning edge-reduction figure.
  - Lewis & Steedman's exact heuristic formula.
  - DeepCubeA journal details (via secondary sources).
  - Venues for NeuroLogic A\*esque (NAACL 2022) and Shi et al. (ICML 2020).
  - Forster et al.'s use of Dijkstra (secondary snippet).
  - GENRE's trie details (from knowledge).
- **Beyond the Beam:** only App. B and §4 were read. Table 11 latency and the Tools/Toys budget curves were truncated.
- **Inconsistencies noted in the sources:**
  - CSV-Decode certification rate: 96.4% vs 98.2%.
  - 2606.04860: node-expansion reductions differ between the abstract and the body.
