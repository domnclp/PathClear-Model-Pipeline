"""Run an unmodified YOLO-FastestV2 script under the .venv-espdet interpreter.

The 2022 code pins torch 1.9 / numpy 1.19 (no Python 3.11 wheels) and uses NumPy aliases removed
since 1.24. This runner restores only those aliases, then executes the script from the pinned clone
(YOLOFASTEST_ROOT) as __main__ with the given arguments; the clone itself stays unmodified.
Usage: run_yolofastest.py <script.py> [script args...]
"""
import os
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(os.environ.get("YOLOFASTEST_ROOT", "C:/Projects/Yolo-FastestV2")).resolve()
for alias, value in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, alias):
        setattr(np, alias, value)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = ROOT / sys.argv[1]
    os.chdir(ROOT)  # The backbone loads ./model/backbone/backbone.pth relative to the clone.
    sys.path.insert(0, str(ROOT))
    # The original never seeds; seed 42 matches every other candidate without editing the script.
    import random
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # torch 1.9 allowed integer tensors to clamp_ against float bounds (utils/loss.py build_target);
    # torch >= 1.12 raises. Casting the bounds to the tensor's dtype reproduces the 1.9 result exactly.
    original_clamp_ = torch.Tensor.clamp_

    def clamp_(self, min=None, max=None):
        if not self.is_floating_point():
            def cast(v):
                return v.to(self.dtype) if torch.is_tensor(v) else (None if v is None else int(v))
            min, max = cast(min), cast(max)
        return original_clamp_(self, min, max)

    torch.Tensor.clamp_ = clamp_
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
