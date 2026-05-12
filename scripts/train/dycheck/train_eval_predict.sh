scene_name="creeper"
time_freq=10
# feature_amplify=5
# max_keypoints=200
# adaptive_points_num=300
nearest_num=6
# max_time=1.0
position_lr_max_steps=40000
data_device="cuda"  # cpu if you run out of memory
step_opacity_iteration=5000
ratio=0.5  # 1.0 for full-sized imagery, 0.5 for downsized

source_path="/home/aneesa/test_data/${scene_name}/splits_gp/"
gcn_source="/home/aneesa/test_data/${scene_name}/splits_gp_predict/"

for feature_amplify in 1 10; do
    for max_keypoints in 100 200; do
        adaptive_points_num=$((max_keypoints + 100))
        model_path="./results/dycheck_1.0_2x/${scene_name}/points_${max_keypoints}_${adaptive_points_num}_amp_${feature_amplify}"
        echo "Starting $model_path..."

        # Train
        CUDA_VISIBLE_DEVICES=1 python train.py -s $source_path \
            -m $model_path --max_points $max_keypoints --adaptive_points_num $adaptive_points_num  \
            --iterations 70000 --test_iterations 70000 --jointly_iteration 1000  --time_freq $time_freq\
            --densify_from_iter 5000 --save_iterations 29998 70000 --checkpoint_iterations 29998 70000 \
            --densify_until_iter 15000  --opacity_reset_interval 3000000 --max_time 1.0 --norm_rotation --step_opacity \
            --position_lr_max_steps $position_lr_max_steps --data_device $data_device --step_opacity_iteration $step_opacity_iteration \
            --eval --use_time_decay --nearest_num $nearest_num --adaptive_interval 1000 --feature_amplify $feature_amplify --ratio $ratio

        # Eval
        CUDA_VISIBLE_DEVICES=1 python eval.py -m $model_path --max_points $max_keypoints --time_freq $time_freq \
            --adaptive_points_num $adaptive_points_num --ckpt_iteration 70000 --feature_amplify $feature_amplify \
            --nearest_num $nearest_num --max_time 1.0 --norm_rotation --step_opacity --ratio $ratio

        # Predict
        max_time=0.835  # index 300+ used for prediction
        noise_init=0.5
        R_ratio=1.0

        num_stage=16
        input_size=10
        noise_step=500
        epoch=1501
        linear_size=128
        exp_name="input${input_size}_stage${num_stage}_hidden${linear_size}_noise${noise_init}_RNoise${R_ratio}_noiseStep${noise_step}_epoch${epoch}"

        CUDA_VISIBLE_DEVICES=1 python train_GCN.py -s $gcn_source -m $model_path --max_points $max_keypoints --time_freq $time_freq \
            --adaptive_points_num $adaptive_points_num --ckpt_iteration 70000 --max_time $max_time \
            --num_stage $num_stage --noise_step $noise_step --noise_init $noise_init --linear_size $linear_size --input_size $input_size \
            --Rscale_ratio $R_ratio --batch_size 32 --epoch $epoch --exp_name $exp_name --predict_more --metrics --norm_rotation \
            --step_opacity --cam_id=-15 --ratio $ratio
    done
done