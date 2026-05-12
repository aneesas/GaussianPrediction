scene_name="creeper"
time_freq=10
feature_amplify=5
max_keypoints=200
adaptive_points_num=300
nearest_num=6
max_time=1.0
model_path="./results/dycheck_${max_time}_2x/${scene_name}/points_200_300"
ratio=0.5  # 1.0 for full-sized imagery, 0.5 for downsized

CUDA_VISIBLE_DEVICES=1 python eval.py -m $model_path --max_points $max_keypoints --time_freq $time_freq \
    --adaptive_points_num $adaptive_points_num --ckpt_iteration 70000 --feature_amplify $feature_amplify \
    --nearest_num $nearest_num --max_time $max_time --norm_rotation --step_opacity --ratio $ratio