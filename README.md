# SunlabParallelNN
Recreating the algorithmic innovations of ParallelNN through CPU and GPU implementations on Sunlab Architecture 

# ParallelNN: CPU + GPU KNN Acceleration  
Final project — CSE 375, Lehigh University

## Overview
This project implements a CPU and GPU parallel nearest‐neighbor (KNN) search inspired by the **ParallelNN** paper.  
We focus on algorithmic innovations, not hardware—using:

- CPU brute-force KNN
- CPU sequential octree-based KNN
- CPU parallel KNN (OpenMP/TBB)
- GPU brute-force KNN (baseline)
- GPU spatial-structure KNN (stretch goal)

Datasets:
- KITTI LiDAR scans
- Synthetic random 3D point clouds

Hardware tested:
- Intel CPUs in SUNLAB
- NVIDIA RTX 3060 (12GB)

## Repository Structure
- `cpu/` — all CPU implementations
- `gpu/` — CUDA kernels + GPU code
- `data/` — KITTI + synthetic data
- `scripts/` — dataset downloads + experiments
- `experiments/` — logs, CSVs, figures
- `report/` — project report + slides

## Build

