# PI + world-aligned canvas: addressed memory in recursion (evaluation only), preregistration

2026-10-15 evening. Written before PI has trained and before any canvas rollout. One change relative to PI; no training.

## Question

Does an explicit world-aligned memory, read at the world's own next-view address, recover revisited terrain in imagination
without harming anything else?

## Evidence that motivated it

- **R16:** after 3+ scrolls, revisited terrain is absent from the screen-slot belief.
- **R17:** the last real token seen at a world position, addressed with the label-free scroll track, reads revisited entering
  terrain at 0.999 on training windows. On panel roots (bright), known entering cells within the horizon read at
  1.000 / 1.000 / 0.987 by age.
- **E18 (`fcanvas`)** failed because its read used the wrong address (0 / 2,018). PI's hard scroll supplies the next-view
  address.

## Arms

Both arms use the same trained PI checkpoints (`pi_seed{1,2}/pi_u3000.pt`), roots, context, actions, H16, 8 samples, and
noise / Gumbel stream.
- **pi_s{1,2}:** `transport_recursion.py`.
- **pi_canvas_s{1,2}:** `pi_canvas.py`. Map cells entering the view whose world position was seen copy the stored token:
  - the address follows the label-free scroll estimate over the real context, then PI's own hard scroll;
  - every frame writes its non-player map tokens.

## Readings (bright roots, both seeds)

- **C1 (primary):** tile error on REVISIT entering cells (`entering_memory.py`, `bright.REVISIT.tile_error`, depths 1-16
  pooled, camera-consistent trajectories). pi_canvas ≤ 0.5 × pi.
- **C2 (no harm):**
  - seen-terrain error at depth 16 ≤ pi + 0.5pp;
  - max phantoms per frame ≤ pi + 0.02;
  - camera consistency at depth 16 ≥ pi − 2pp.
- **Verdict:**
  - **"Addressed world-aligned memory works in recursion"** if C1 and C2 hold at both seeds.
  - **"Memory recovers terrain but harms the rollout"** if C1 holds and C2 fails.
  - **"Not supported"** if C1 fails.
- **Secondary:** NEW entering error (should be unchanged), cow / zombie census, persisting-mob score, decision proxies, share
  of entering cells read from memory.

## Known limits

- The canvas also stores mobs. A revisited position replays the last mob seen there, with no off-screen dynamics. Census and
  phantoms are reported for that reason.
- The canvas is outside the backbone: the screen-slot belief still does not hold the remembered content. The gate and chance
  operator see it only through the context frames.
- If PI's own scroll decisions are poor, the canvas misaddresses the view. Camera consistency is reported, and C1 counts only
  camera-consistent trajectories.
- Fresh confirmation stays sealed.
