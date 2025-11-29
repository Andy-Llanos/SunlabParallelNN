import numpy as np
import sys

N = int(sys.argv[1])
out = sys.argv[2]

pts = np.random.rand(N, 3).astype(np.float32)
pts.tofile(out)

print(f"Wrote {N} synthetic points to {out}")
