scene_name="creeper"
time_freq=10
max_keypoints=200  # TODO try changing this/below
adaptive_points_num=300
max_time=0.835  # index 300+ used for prediction
noise_init=0.5
R_ratio=1.0

num_stage=16
input_size=10
noise_step=500
epoch=1501
linear_size=128
ratio=0.5  # 1.0 for full-sized imagery, 0.5 for downsized
exp_name="input${input_size}_stage${num_stage}_hidden${linear_size}_noise${noise_init}_RNoise${R_ratio}_noiseStep${noise_step}_epoch${epoch}"

gcn_source="/home/aneesa/test_data/${scene_name}/splits_gp_predict/"
model_path="./results/dycheck_1.0_2x/${scene_name}/points_200_300/"  # experiment with max_time=1.0 actually only trains on first 300 frames

CUDA_VISIBLE_DEVICES=1 python train_GCN.py -s $gcn_source -m $model_path --max_points $max_keypoints --time_freq $time_freq \
    --adaptive_points_num $adaptive_points_num --ckpt_iteration 70000 --max_time $max_time \
    --num_stage $num_stage --noise_step $noise_step --noise_init $noise_init --linear_size $linear_size --input_size $input_size \
    --Rscale_ratio $R_ratio --batch_size 32 --epoch $epoch --exp_name $exp_name --predict_more --metrics --norm_rotation \
    --step_opacity --cam_id=-15 --ratio $ratio