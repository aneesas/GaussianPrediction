import json
import os

SCENE_NAME = "jumpingjacks"
DATA_DIR = "./datasets/d-nerf/data"
WINDOW = 20
K_VALUES = [6]

src = os.path.join(DATA_DIR, SCENE_NAME, "transforms_train.json")
with open(src) as f:
    original = json.load(f)

frames = original["frames"]
total = len(frames)

def removed_indices(k, total, window):
    indices = set()
    for start in range(0, total, window):
        for i in range(k):
            if start + i < total:
                indices.add(start + i)
    return indices

for k in K_VALUES:
    drop = removed_indices(k, total, WINDOW)
    kept = [frame for i, frame in enumerate(frames) if i not in drop]
    out = {**original, "frames": kept}

    out_dir = os.path.join(DATA_DIR, f"{SCENE_NAME}_rm{k}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transforms_train.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"rm{k}: removed {len(drop)} frames, kept {len(kept)}/{total} → {out_path}")
