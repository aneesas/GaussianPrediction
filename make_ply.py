"""Build points3D_downsample.ply for creeper from iPhone LiDAR depth, as an
alternative to running COLMAP. Back-projects depth + RGB of N evenly-spaced
frames into world coordinates and voxel-downsamples to ~40K points."""

import json
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import storePly

DATASET_DIR = "/home/aneesa/test_data/creeper"
OUT_PATH = os.path.join(DATASET_DIR, "splits_gp", "points3D_downsample.ply")

FRAME_STRIDE = 30          # every 30th frame from training pool [0..299] -> 10 frames
TARGET_POINTS = 40000      # voxel-downsample target

def unproject(depth, rgb, cam):
    H, W = depth.shape[:2]
    fx = fy = cam["focal_length"]
    cx, cy = cam["principal_point"]

    v_grid, u_grid = np.mgrid[:H, :W]
    z = depth[..., 0]
    mask = z > 0

    u = u_grid[mask].astype(np.float32)
    v = v_grid[mask].astype(np.float32)
    z = z[mask]

    # Camera-space (COLMAP convention: y-down, z-forward)
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)

    # Camera -> world. HyperNeRF/DyCheck convention:
    #   orientation = w2c rotation, position = camera world center
    # so c2w rotation = orientation.T, c2w translation = position
    orientation = np.array(cam["orientation"])
    position = np.array(cam["position"])
    pts_world = pts_cam @ orientation + position

    cols = rgb[mask]
    return pts_world, cols

def main():
    with open(os.path.join(DATASET_DIR, "dataset.json")) as f:
        ids = json.load(f)["ids"]
    with open(os.path.join(DATASET_DIR, "metadata.json")) as f:
        meta = json.load(f)
    ids_sorted = sorted(ids, key=lambda fid: meta[fid]["warp_id"])
    selected = ids_sorted[:300:FRAME_STRIDE]
    print(f"Unprojecting {len(selected)} frames: {selected}")

    all_pts, all_cols = [], []
    for fid in selected:
        depth = np.load(os.path.join(DATASET_DIR, "depth", "1x", f"{fid}.npy"))
        rgb = np.asarray(Image.open(os.path.join(DATASET_DIR, "rgb", "1x", f"{fid}.png")))[..., :3]
        with open(os.path.join(DATASET_DIR, "camera", f"{fid}.json")) as f:
            cam = json.load(f)
        pts, cols = unproject(depth, rgb, cam)
        all_pts.append(pts)
        all_cols.append(cols)
        print(f"  {fid}: {len(pts):,} points")
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    print(f"Total before downsample: {len(pts):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    v = 0.02
    while len(pcd.points) > TARGET_POINTS:
        pcd = pcd.voxel_down_sample(voxel_size=v)
        v += 0.01
    print(f"After downsample: {len(pcd.points):,} points (voxel size {v - 0.01:.2f})")

    out_pts = np.asarray(pcd.points)
    out_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    storePly(OUT_PATH, out_pts, out_cols)
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
