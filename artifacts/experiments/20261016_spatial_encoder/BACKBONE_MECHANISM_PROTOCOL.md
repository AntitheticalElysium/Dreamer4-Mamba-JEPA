# Backbone mechanism check — PROTOCOL (2026-10-17; written before any number; read-only, saved checkpoints)

**User mandate.** Run the cheap backbone-gradient measurement. If it is negative, close the encoder question unless a concrete
suspicion remains, and return to dynamics.

**The fact to explain (R68).** MoP-s1's backbone-LN tokens have the same static mob decodability as Raw-s1's (0.986 vs 0.987;
real-next 0.91 vs 0.93). Yet an identical D world trained on them learns **camera-transformed persistence** worse at 12k
(−0.152 [−0.206, −0.102]) and **mob motion** worse at 36k (−0.079 [−0.144, −0.019]).

**The concrete suspicion,** grounded in D's mechanism (`tworld.py` corrt head):
- D predicts each next token as a softmax mixture of the token's own copy, its **four grid neighbours' copies** and one generated
  token, with a shared "did the view move" logit.
- After a camera shift, persistence is achieved by copying the neighbour token.
- That works only if a token's content code is **translation-consistent**: the same object gives a similar token at the adjacent
  cell. If MoP's tokens carry more **absolute-position** information, a copied neighbour is the wrong token. Static per-cell probes
  would not see this, because they are fitted per cell position over pooled data.

## Part A — representation property, measured directly (no gradients)

**Panel.**
- All diagnosis roots; the factual future (f = 0), consecutive frames j → j+1 up to 16 steps, alive.
- Exact camera shift from exact tiles (`objective_audit.shift_of`), with `sources()` mapping each next cell to its world source.
- **Persisting cells:** source in view, same tile **and** same mob layer at source and destination, neither of them the player cell
  (31).

**Metrics,** per encoder, on LN(last-layer patch tokens):
- **A1 transport ratio (camera-moved pairs):** Σ‖z_{j+1}[c] − z_j[src(c)]‖² / Σ‖z_{j+1}[c] − z_j[c]‖² over persisting cells.
  - It compares world-aligned copy error with screen copy error.
  - Lower means tokens move with their content (translation-consistent).
- **A2 static drift (camera-static pairs):** Σ‖z_{j+1}[c] − z_j[c]‖² over persisting cells ÷ the total token variance.
- **A3 position share:** the share of map-token variance explained by absolute cell position **within** content classes:
  - group means by (tile, mob bits, cell) vs by (tile, mob bits), all panel frames;
  - higher means more absolute-position information.

**Encoders:**
- Raw-s1 (`raw@10000`);
- MoP-s1 backbone at 500, 1k, 2k, 4k and 10k (the trajectory across the decline);
- MoP-s2 at 10k; full s1 / s2 at 10k (context).

**Intervals:** episode-cluster bootstrap, 2,000 draws, paired (the same units).

## Part B — which training gradients reached the ViT backbone (MoP-s1 and MoP-s2 at 1k, 2k, 4k, 10k)

- **Exact replication of one training step's loss** (`spatial_encoder.py`):
  cls_pred + 0.09·cls_reg + patch min-K MSE + 0.1·router CE + 0.0225·patch SIGReg.
  - Train-mode modules; the full B128 × T4 cohort in one projector call; bf16 autocast as in training.
  - 4 fixed JointSampler batches (seed 20261110), the same for every checkpoint.
  - Fixed SIGReg generators.
- **Per-term gradients** w.r.t. every ViT backbone parameter (`bundle.encoder`), via `autograd.grad`. The patch MSE is split into
  its **input path** (targets detached) and **target path** (inputs detached); their sum is the full term.
- **Shares of the backbone gradient:**
  - additive projection ⟨g_t, g_tot⟩ / ‖g_tot‖² (sums to 1), norm ratio and cosine;
  - computed on raw gradients **and** on Adam-preconditioned gradients g_t / (√v̂ + ε), using each checkpoint's saved AdamW second
    moments (the update direction actually taken, up to clipping, which scales all terms equally);
  - by parameter group: patch embedding + positional embedding, blocks 1-4, 5-8, 9-12, final norm.
- **First-order effect** of each term's preconditioned update direction −u_t on the Part A properties, computed differentiably on a
  fixed panel (256 camera-moved transitions; 512 map frames):
  - A1: transport ratio;
  - A3 proxy: within-tile position variance on the panel;
  - a static mob-probe loss (a ridge probe fitted at that checkpoint, held fixed).
  - Reported as −⟨∇M, u_t⟩, normalized by the total. A positive value means the term's update **increases** M (more
    position-dependent / less transportable / less decodable).

## Pre-declared readings

**Negative (close the encoder question):** both of the following hold. Then translation consistency is not the mechanism, and no
single term is implicated.
1. MoP-s1 at 10k vs Raw shows no material difference on A1 and A3 (paired interval contains 0, or |difference| < 10% of Raw's
   value).
2. Part B finds no term whose preconditioned update consistently increases A1 or A3 (the same sign at both seeds and at ≥ 3 of 4
   checkpoints).

**Concrete suspicion (report; no training without the user):** MoP-s1 is materially worse on A1 or A3 (interval excludes 0 and
|difference| ≥ 10%), **and** Part B identifies a term with a consistent harmful first-order effect.

**Anything else:** reported as is, as weak evidence. It does not justify more encoder work by itself.

## Limits
- First-order effects describe local update directions at saved checkpoints, not integrated training history.
- A1–A3 are proxies chosen from D's copy mechanism; they do not measure D's learning directly.
- One encoder seed pair for the matched downstream result.
