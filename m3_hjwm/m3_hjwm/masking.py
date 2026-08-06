from __future__ import annotations
import torch
from torch import Tensor


def multi_block_mask(
    batch: int,
    grid_h: int,
    grid_w: int,
    ratio: float,
    num_blocks: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a boolean target mask [B, H, W].

    This is a compact multi-block masker for small environments. Blocks are sampled
    until approximately `ratio` of the grid is covered. The context encoder receives
    the complement. Unlike the previous project, masking occurs on an 8x8 grid rather
    than after globally pooling a 4x4 map.
    """
    target = torch.zeros(batch, grid_h, grid_w, dtype=torch.bool, device=device)
    target_count = max(1, int(round(grid_h * grid_w * ratio)))
    for b in range(batch):
        attempts = 0
        while int(target[b].sum()) < target_count and attempts < num_blocks * 8:
            # Bias towards substantial blocks, as I-JEPA does.
            bh = int(torch.randint(max(1, grid_h // 4), max(2, grid_h * 3 // 4 + 1), (1,), generator=generator, device=device))
            bw = int(torch.randint(max(1, grid_w // 4), max(2, grid_w * 3 // 4 + 1), (1,), generator=generator, device=device))
            y = int(torch.randint(0, grid_h - bh + 1, (1,), generator=generator, device=device))
            x = int(torch.randint(0, grid_w - bw + 1, (1,), generator=generator, device=device))
            target[b, y:y + bh, x:x + bw] = True
            attempts += 1
        if int(target[b].sum()) < target_count:
            flat = torch.randperm(grid_h * grid_w, generator=generator, device=device)[:target_count]
            target[b].view(-1)[flat] = True
    return target
