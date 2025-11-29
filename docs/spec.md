## Project Specification (Frozen on Day 2)

- **Problem:** k-nearest neighbor (kNN) search on 3D point clouds.
- **Dimension:** fixed to 3D (x, y, z).
- **Distance metric:** Euclidean (L2).
- **k:** small, fixed (e.g., k = 16).
- **Dataset:**
  - Synthetic random 3D points in [0, 1]^3.
  - KITTI LiDAR frames (converted to 3D points).
- **CPU Subproject:**
  - Brute-force kNN.
  - Sequential octree build.
  - Sequential octree kNN.
  - Parallel octree kNN over queries using OpenMP or TBB.
- **GPU Subproject:**
  - Brute-force kNN with one thread per query.
  - (Stretch goal) Octree or grid-based GPU kNN.

- **Hardware:**
  - SUNLAB CPU nodes (exact model TBD).
  - NVIDIA GeForce RTX 3060 (12 GB) in SUNLAB.

- **Deliverables:**
  - Correctness: GPU & CPU results match on small test cases.
  - Performance: runtime + speedup plots for:
    - CPU brute-force,
    - CPU octree (seq + parallel),
    - GPU brute-force.
  - Report + slides linking design to ParallelNN paper.
