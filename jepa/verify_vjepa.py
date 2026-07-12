"""Step-1 verification: load frozen V-JEPA 2.1 ViT-B/16 and confirm we can encode a
single frame (image mode), the token/feature shape, real param count, VRAM, latency.

Source facts verified in the vjepa2 repo first:
- ViT.forward accepts a 4-D (B,C,H,W) tensor -> T=1 image mode (uses RoPE).
- Weights live at dl.fbaipublicfiles.com (repo shipped a localhost placeholder URL,
  which we fixed in our clone). We import the builder directly to avoid hubconf's
  heavy data deps (decord/webdataset).
"""
import sys
import time
import torch

VJEPA_REPO = "/tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/f3c69503-d8a3-430b-9456-5002f1f83805/scratchpad/vjepa2"
sys.path.insert(0, VJEPA_REPO)


def main():
    dev = "cuda"
    from src.hub.backbones import vjepa2_1_vit_base_384
    print("building ViT-B (checkpoint cached after first run)...")
    obj = vjepa2_1_vit_base_384(pretrained=True)
    if isinstance(obj, (tuple, list)):
        print(f"builder returned {len(obj)} objects: {[type(o).__name__ for o in obj]}")
        enc = obj[0]
    else:
        enc = obj
    enc = enc.to(dev).eval()
    n = sum(p.numel() for p in enc.parameters()) / 1e6
    print(f"ENCODER params: {n:.1f}M  embed_dim={getattr(enc,'embed_dim','?')} "
          f"patch={getattr(enc,'patch_size','?')} use_rope={getattr(enc,'use_rope','?')}")
    weight_mb = sum(p.numel() * p.element_size() for p in enc.parameters()) / 1e6
    print(f"encoder weights: {weight_mb:.0f} MB")

    # patch_embed is Conv3d w/ tubelet_size=2 -> input is a mini-clip (B,C,T,H,W), T even.
    # A "per-frame" latent = replicate the frame to T=2 -> 1 temporal token -> spatial tokens only.
    for (T, res) in [(2, 384), (2, 256), (2, 224), (16, 256)]:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, 3, T, res, res, device=dev)
        try:
            with torch.no_grad():
                t0 = time.time()
                y = enc(x)
                torch.cuda.synchronize()
                dt = (time.time() - t0) * 1000
            shape = tuple(y.shape) if torch.is_tensor(y) else [tuple(t.shape) for t in y]
            exp = (T // 2) * (res // 16) ** 2
            print(f"T={T:2d} res={res:3d}  out={shape}  (expect {exp} tokens)  {dt:6.1f} ms  "
                  f"peakVRAM={torch.cuda.max_memory_allocated()/1e6:.0f}MB")
        except Exception as e:  # noqa: BLE001
            print(f"T={T:2d} res={res:3d}  FAILED {type(e).__name__}: {str(e)[:140]}")
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
