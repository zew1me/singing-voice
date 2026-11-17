"""Download every Seed-VC weight the worker needs, by loading the engine once (build time, CPU).

Run from the image build so cold starts load from disk instead of Hugging Face.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runpod_worker.engine import SeedVCEngine  # noqa: E402

if __name__ == "__main__":
    engine = SeedVCEngine(os.environ.get("SEED_VC_HOME", "/opt/seed-vc"), fp16=False)
    print("prefetched", engine.describe())
