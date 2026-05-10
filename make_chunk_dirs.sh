# scene_name=jumpingjacks
# base=./datasets/d-nerf/data/${scene_name}

# for k in 6; do
#   dir="${base}_rm${k}"
#   mkdir -p "$dir"
#   ln -s ../${scene_name}/transforms_test.json "$dir/transforms_test.json"
#   ln -s ../${scene_name}/transforms_val.json  "$dir/transforms_val.json"
#   ln -s ../${scene_name}/points3d.ply         "$dir/points3d.ply"
#   ln -s ../${scene_name}/train                "$dir/train"
#   ln -s ../${scene_name}/test                 "$dir/test"
# done

scene_name=jumpingjacks
dir=./datasets/d-nerf/data/${scene_name}_periodic

ln -s ../${scene_name}/transforms_test.json "$dir/transforms_test.json"
ln -s ../${scene_name}/transforms_val.json  "$dir/transforms_val.json"
ln -s ../${scene_name}/points3d.ply         "$dir/points3d.ply"
ln -s ../${scene_name}/train                "$dir/train"
ln -s ../${scene_name}/test                 "$dir/test"
