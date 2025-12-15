import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_kitti.csv")

# Example 1: bar chart of speedup per method for one config
cfg = df[(df["N"] == 100000) & (df["Q"] == 2000) & (df["k"] == 16)]

plt.figure()
plt.bar(cfg["method"], cfg["speedup_vs_cpuBF"])
plt.ylabel("Speedup vs CPU brute-force")
plt.title("Synthetic KITTI-like, N=100k, Q=2k, k=16")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("speedup_methods_N100k_Q2k_k16.png", dpi=200)

# Example 2: speedup vs N for GPU and CPU TBB (if you vary N)
gpu = df[df["method"] == "gpu_bf"]
oct_tbb = df[df["method"] == "cpu_octree_tbb"]

plt.figure()
plt.plot(gpu["N"], gpu["speedup_vs_cpuBF"], "o-", label="GPU brute-force")
plt.plot(oct_tbb["N"], oct_tbb["speedup_vs_cpuBF"], "s-", label="CPU octree TBB")
plt.xlabel("Number of points (N)")
plt.ylabel("Speedup vs CPU brute-force")
plt.title("Speedup vs N (Synthetic KITTI-like)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("speedup_vs_N.png", dpi=200)
