"""
Generate dense depth maps from KITTI-360 LiDAR data.
Usage: python generate_depth_maps.py --data_root /path/to/kitti360 --output_dir depth_maps
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

def get_calibration():
    Tr_cam_to_velo = np.array([
        [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
        [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
        [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
        [0.0, 0.0, 0.0, 1.0]
    ])
    Tr_velo_to_cam = np.linalg.inv(Tr_cam_to_velo)
    K = np.array([[552.554261, 0.0, 682.049453],
                  [0.0, 552.554261, 238.769549],
                  [0.0, 0.0, 1.0]])
    R_rect = np.eye(4)
    R_rect[:3, :3] = np.array([[0.999974, -0.007141, -0.000089],
                               [0.007141, 0.999969, -0.003247],
                               [0.000112, 0.003247, 0.999995]])
    Tr_velo_to_rect = R_rect @ Tr_velo_to_cam
    return K, Tr_velo_to_rect

def project_points(points, K, Tr, width=1408, height=376, max_depth=80.0, min_depth=0.5):
    pts_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    pts_cam = (Tr @ pts_hom.T).T
    valid = (pts_cam[:, 2] > min_depth) & (pts_cam[:, 2] <= max_depth)
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return None, None, None
    uv_hom = K @ pts_cam[:, :3].T
    depth = uv_hom[2, :]
    u = np.round(uv_hom[0, :] / depth).astype(np.int32)
    v = np.round(uv_hom[1, :] / depth).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_bounds], v[in_bounds], depth[in_bounds]

def generate_depth_maps(data_root, output_dir, sequences, max_depth=80.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "numpy").mkdir(exist_ok=True)
    K, Tr = get_calibration()
    total_frames = 0
    for seq in sequences:
        seq_name = f"2013_05_28_drive_{int(seq):04d}_sync"
        lidar_dir = Path(data_root) / "3D Lidar" / "download_3d_velodyne" / "KITTI-360" / "data_3d_raw" / seq_name / "velodyne_points" / "data"
        if not lidar_dir.exists():
            print(f" LiDAR directory not found: {lidar_dir}")
            continue
        bin_files = sorted(lidar_dir.glob("*.bin"))
        for bin_file in tqdm(bin_files, desc=f"Sequence {seq}"):
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            u, v, depth = project_points(points, K, Tr, max_depth=max_depth)
            if u is None:
                continue
            depth_map = np.zeros((376, 1408), dtype=np.float32)
            for ui, vi, di in zip(u, v, depth):
                if depth_map[vi, ui] == 0 or di < depth_map[vi, ui]:
                    depth_map[vi, ui] = di
            
            frame_id = bin_file.stem
            np.save(output_dir / "numpy" / f"{seq}_{frame_id}.npy", depth_map)
            total_frames += 1
    print(f"Done. Generated {total_frames} depth maps in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="depth_maps")
    parser.add_argument("--sequences", nargs="+", default=["0000","0002","0003","0005","0006","0007","0009","0010"])
    parser.add_argument("--max_depth", type=float, default=80.0)
    args = parser.parse_args()
    generate_depth_maps(args.data_root, args.output_dir, args.sequences, args.max_depth)
