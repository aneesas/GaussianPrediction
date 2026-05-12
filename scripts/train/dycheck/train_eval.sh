echo "Start training and evaluation process"
echo "Please make sure your scripts [max_time] is 1.0 "

read -p "Do you want to continue? (yes/no): " user_input

if [ "$user_input" = "yes" ]; then
    echo "Continuing execution..."
else
    echo "Exiting..."
    exit 1
fi

./scripts/train/dycheck/creeper.sh 
wait
./scripts/eval/dycheck/creeper.sh 
wait

root="./results"
model_path="points_200_300"
dataset="dycheck_1.0_2x"

python show.py -r $root -d $dataset -m $model_path
