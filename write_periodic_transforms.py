import json
import os

SCENE_NAME = "bouncingballs"
DATA_DIR = "./datasets/d-nerf/data"
PERIOD = 1.0
HALF = PERIOD / 2

src = os.path.join(DATA_DIR, SCENE_NAME, "transforms_train.json")
with open(src) as f:
    original = json.load(f)

frames = original["frames"]

extra = []
for frame in frames:
    t = float(frame["time"])
    if 0 < t <= HALF:
        new_frame = dict(frame)
        new_frame["time"] = t + PERIOD
        extra.append(new_frame)

combined = frames + extra
out = {**original, "frames": combined}

out_dir = os.path.join(DATA_DIR, f"{SCENE_NAME}_periodic")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "transforms_train.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"original: {len(frames)} frames, added {len(extra)} duplicates → {len(combined)} total")
print(f"time range: [{min(fr['time'] for fr in combined):.4f}, {max(fr['time'] for fr in combined):.4f}]")
print(f"written to: {out_path}")
