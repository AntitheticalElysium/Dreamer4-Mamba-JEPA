"""Hazard-choice states: root, a safe action, and a damaging action, with y."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax

import replay

NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
         "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICK",
         "MAKE_STONE_PICK", "MAKE_IRON_PICK", "MAKE_WOOD_SWORD",
         "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"]
SCALE, LABEL, PAD = 5, 14, 4


def font(size=10):
    for name in ("/usr/share/fonts/TTF/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


F, T = font(10), font(12)


def tile(frame, label, colour=(230, 230, 230)):
    body = Image.fromarray(frame).resize((63 * SCALE, 63 * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (63 * SCALE, 63 * SCALE + LABEL), (18, 18, 22))
    canvas.paste(body, (0, LABEL))
    ImageDraw.Draw(canvas).text((2, 1), label, fill=colour, font=F)
    return canvas


def border(image, colour, width=3):
    out = Image.new("RGB", (image.width + 2 * width, image.height + 2 * width), colour)
    out.paste(image, (width, width))
    return out


records = torch.load(HERE / "nonterminal_forks.pt", weights_only=False)
both = [r for r in records
        if (np.array(r["health"]) <= -1).any() and (np.array(r["health"]) >= 0).any()]
both.sort(key=lambda r: -abs(np.array(r["health"]).min()))
chosen = both[:2] + both[len(both) // 3 : len(both) // 3 + 2] + both[-2:]

_, _, _, step_fn, frame_fn = replay.env_and_render()
KEY = jax.random.PRNGKey(7)
rows = []
for r in chosen:
    state = replay.advance_to(r["shard"], r["slot"], r["t"])
    health = np.array(r["health"])
    y = np.array(r["y"])
    hurt = int(np.argmin(health))
    safe = int(np.argmax(np.where(health >= 0, y * 0 + np.arange(17) * 0 + 1, -1)))
    safe = int(np.where(health >= 0)[0][0])
    root_frame = np.asarray(frame_fn(state))
    tiles = [border(tile(root_frame,
                         f"ROOT hp{r['base_health']:.0f} ach{r['base_ach']} "
                         f"zombies{r['zombies']}", (150, 190, 240)), (80, 110, 160))]
    for action, kind in ((safe, "SAFE"), (hurt, "DAMAGE")):
        _, nxt, _, _, _ = step_fn(KEY, state, int(action))
        tiles.append(border(
            tile(np.asarray(frame_fn(nxt)),
                 f"{kind} {NAMES[action][:12]}  dHP{health[action]:+.0f}  y={y[action]:+.3f}",
                 (200, 230, 200) if kind == "SAFE" else (255, 150, 140)),
            (60, 120, 80) if kind == "SAFE" else (170, 70, 60)))
    rows.append(tiles)

width = max(t.width for row in rows for t in row)
height = max(t.height for row in rows for t in row)
canvas = Image.new("RGB", (3 * (width + PAD) + PAD, len(rows) * (height + PAD) + PAD + 22),
                   (18, 18, 22))
ImageDraw.Draw(canvas).text(
    (PAD, 5),
    "Hazard-choice states: the action decides whether the player is hit -- but the "
    "fatality direction y barely moves",
    fill=(240, 240, 240), font=T)
for i, row in enumerate(rows):
    for j, t in enumerate(row):
        canvas.paste(t, (PAD + j * (width + PAD), 22 + PAD + i * (height + PAD)))
canvas.save(HERE / "hazard_choice.png")
print("wrote hazard_choice.png", canvas.size)
for r in chosen:
    health = np.array(r["health"]); y = np.array(r["y"])
    print(f"  band {r['band']} eps {r['epsilon']}  hp {r['base_health']:.0f}  "
          f"minΔhp {health.min():+.0f}  y(damage) {y[int(np.argmin(health))]:+.3f}  "
          f"y(safe) {y[int(np.where(health>=0)[0][0])]:+.3f}")
