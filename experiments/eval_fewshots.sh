#!/bin/bash
model_path=${1}
save_log_name=${2}
save_log_pth="${save_log_name}/log_$(date +%Y%m%d_%H%M%S).out"

mkdir -p "$(dirname "${save_log_pth}")"
> "${save_log_pth}"
exec &> >(tee -a ${save_log_pth})

echo "Eval logging path: $save_log_pth"
echo "Model path: $model_path"

echo "[START] - Start Evaluating Model"
python ../utils/hf_eval_fewshots.py \
  --model_paths ${model_path} \
  --eval_device cuda \
  --save_log_name ${save_log_name} \
  --output_dir ${save_log_name}

echo "[FINISH] - Finish Evaluating Model"
