import json
import os
import random

DATASET_DIR = "/home/aneesa/test_data/creeper"
OUT_DIR = os.path.join(DATASET_DIR, "splits_gp")
SEED = 42

TOTAL = 360
TRAIN_POOL = 300
VAL_SIZE = 30
TEST_SIZE = 30

with open(os.path.join(DATASET_DIR, "dataset.json")) as f:
    original = json.load(f)
ids = original["ids"]
assert len(ids) == TOTAL

with open(os.path.join(DATASET_DIR, "metadata.json")) as f:
    meta = json.load(f)

ids_sorted = sorted(ids, key=lambda fid: meta[fid]["warp_id"])

predict_idx = list(range(TRAIN_POOL, TOTAL))

rng = random.Random(SEED)
shuffled = list(range(TRAIN_POOL))
rng.shuffle(shuffled)
val_idx = sorted(shuffled[:VAL_SIZE])
test_idx = sorted(shuffled[VAL_SIZE:VAL_SIZE + TEST_SIZE])
held_out = set(val_idx) | set(test_idx)
train_idx = [i for i in range(TRAIN_POOL) if i not in held_out]

train_ids = [ids_sorted[i] for i in train_idx]
val_ids = [ids_sorted[i] for i in val_idx]
test_ids = [ids_sorted[i] for i in test_idx]
predict_ids = [ids_sorted[i] for i in predict_idx]

os.makedirs(OUT_DIR, exist_ok=True)

new_dataset = dict(original)
new_dataset["train_ids"] = train_ids
new_dataset["val_ids"] = val_ids
new_dataset["predict_ids"] = predict_ids
new_dataset["num_exemplars"] = len(train_ids)

dataset_path = os.path.join(OUT_DIR, "dataset.json")
with open(dataset_path, "w") as f:
    json.dump(new_dataset, f, indent=2)
print(f"dataset.json: train={len(train_ids)}, val={len(val_ids)}, predict={len(predict_ids)} -> {dataset_path}")

def write_indices(name, indices):
    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(str(i) for i in indices) + "\n")
    print(f"{name}.txt: {len(indices)} indices -> {txt_path}")

write_indices("train", train_idx)
write_indices("val", val_idx)
write_indices("test", test_idx)
write_indices("predict", predict_idx)
