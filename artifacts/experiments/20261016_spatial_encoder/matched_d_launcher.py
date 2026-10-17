"""MATCHED_ENCODER_PREREG.md: run 20260927_levers/tworld.py UNMODIFIED (sha256 2c31cce7...) with one extra registered pool name,
mop_s1 -> artifacts/eda/spatial_pool_mop_s1_v1 (the Raw pool's windows re-encoded by MoP-s1's backbone-LN). Everything else -
arguments, seeds, optimizer, snapshots, output naming (the pool name is part of the run name), the embedded script hash - is
tworld.py's own. Launch from the repository root with TRITON_F32_DEFAULT=ieee PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
as the historical lane did (20260927_levers/lib.sh).
Usage: matched_d_launcher.py <tworld.py arguments>
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260927_levers'))
import tworld as TW                                                            # noqa: E402

VERIFIED = '2c31cce7eff9ceb58a59d3244180656f5d9b3ea36d7be96a9801ae7e860df935'
assert hashlib.sha256(Path(TW.__file__).read_bytes()).hexdigest() == VERIFIED, 'tworld.py differs from the verified file'
TW.POOLS['mop_s1'] = ROOT / 'artifacts/eda/spatial_pool_mop_s1_v1'

if __name__ == '__main__':
    raise SystemExit(TW.main(sys.argv[1:]))
