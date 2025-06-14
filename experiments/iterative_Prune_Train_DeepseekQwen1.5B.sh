
#!/bin/bash
save_dir="$1"
processed_dataset_dir="$2"

start_time=$(date +%s)
num_gpu=4
iterative_prune_train_step=11
repeat_dataset=1
seperate_strategy=linear
warmup_ratio_int=5
# batch_size=32
# micro_batch_size=16
batch_size=32
micro_batch_size=8
gradient_accumulation_steps=$(( ${batch_size} / ${micro_batch_size} ))
lr_scheduler=WSD
learning_rate=2e-5
min_learning_rate=2e-6
max_grad_norm=1
cutoff_len=1024

base_dir="${save_dir}/iterative_prune_train_Deepseek_1.5B_instruct_1_${iterative_prune_train_step}_steps_repeat_${repeat_dataset}_strategy_${seperate_strategy}_${lr_scheduler}_max_lr_${learning_rate}"
save_log_name="${base_dir}/log"
save_log_pth="${save_log_name}/log.out"
tmp_result_name="${base_dir}/tmp_result.out"
pruned_model_save_dir="${base_dir}/pruned_model"
trained_model_save_dir="${base_dir}/trained_model"
output_model_save_dir="${base_dir}/output_model"

calibration_data_path='slimpajama'
model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

original_param_num=1777088000
target_param_num=1005461760

tokenized_dataset_dir="/mnt/beichen/processed_datasets_deepseekQwen1.5B_11_steps_repeat_1_strategy_linear_instruct_1/tokenized_dataset"
num_samples=7714363

mkdir -p "$(dirname "${save_log_pth}")"
exec &> >(tee -a ${save_log_pth})
# tokenized_dataset_dir='/mnt/beichen/processed_datasets_deepseekQwen1.5B_11_steps_repeat_1_strategy_linear_slimpajama_20B/tokenized_dataset'
# num_samples=19428541


echo "Logging path: ${save_log_pth}"
echo "Temporary returned value path: ${tmp_result_name}"
echo "Model path: ${model_path}"
echo "Tokenized dataset path: ${tokenized_dataset_dir}"
echo "Total sample number: ${num_samples}"
echo "Pruned model path: ${pruned_model_save_dir}"
echo "Trained model path: ${trained_model_save_dir}"
echo "Output model path: ${output_model_save_dir}"

echo "Original parameter number: ${original_param_num}"
echo "Target parameter number: ${target_param_num}"
echo "Iterative prune and train steps: ${iterative_prune_train_step}"
echo "Microbatch size: ${micro_batch_size}"
echo "Max learning rate: ${learning_rate}"
echo "Min learning rate: ${min_learning_rate}"
echo "Training on ${num_gpu} GPUs"

# mkdir -p "${base_dir}/gpu_logs"

# cleanup() {
#     echo "Cleaning up GPU monitoring process..."
#     if [ ! -z "$GPU_MONITOR_PID" ]; then
#         pkill -P $GPU_MONITOR_PID
#         kill -9 $GPU_MONITOR_PID 2>/dev/null
#     fi
# }

# trap cleanup SIGINT SIGTERM SIGKILL EXIT SIGQUIT

# (
#     echo $$ > "${base_dir}/gpu_logs/monitor.pid"
#     while true; do
#         date '+%Y-%m-%d %H:%M:%S' >> "${base_dir}/gpu_logs/gpu_usage.log"
#         nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,temperature.gpu --format=csv >> "${base_dir}/gpu_logs/gpu_usage.log"
#         echo "-------------------" >> "${base_dir}/gpu_logs/gpu_usage.log"
#         sleep 60
#     done
# ) &
# GPU_MONITOR_PID=$!

denominator=$(( ${micro_batch_size} * 4 * ${gradient_accumulation_steps} ))
if [ $(( ${num_samples} % ${denominator} )) -ne 0 ]; then
    remainder=1
else
    remainder=0
fi
quotient=$(( ${num_samples} / ${denominator} ))
total_training_steps=$(( ${quotient} + ${remainder} ))
base_training_steps=0
total_warmup_steps=$(( (${warmup_ratio_int} * ${total_training_steps}) / 100 ))

echo "Total prune and train ${iterative_prune_train_step} steps, with ${total_training_steps} training steps and ${total_warmup_steps} warmup steps"


for stage in $(seq 1 ${iterative_prune_train_step}); do
    echo "[START] - Start Pruning Model Stage $stage of $iterative_prune_train_step"
    output_path="${pruned_model_save_dir}/adaptive_prune_stage_${stage}_of_${iterative_prune_train_step}/output_model.bin"
    cur_target_param_num=$((original_param_num - (original_param_num - target_param_num) * stage / iterative_prune_train_step))
    echo "Current step targe pruned model parameter number: ${cur_target_param_num}"
    echo "Pruning Model ${model_path}"

    python ../hf_prune.py \
        --adpative_prune \
        --layer_prune_distribution_amplitude 0.02 \
        --iterative_steps 50 \
        --base_model ${model_path} \
        --calibration_data_path ${calibration_data_path} \
        --pruning_ratio 1.00 \
        --target_param_num ${cur_target_param_num} \
        --device cuda \
        --block_wise \
        --block_mlp_layer_start 0 \
        --block_mlp_layer_end 28 \
        --block_attention_layer_start 0 \
        --block_attention_layer_end 28 \
        --save_log_name ${save_log_name} \
        --output_pth ${output_path} \
        --pruner_type taylor \
        --taylor param_first \
        --taylor_seq_len 64 \
        --num_examples 512 \
        --save_model

    echo "[FINISH] - Finish Pruning Model Stage $stage of $iterative_prune_train_step"
    
    echo "[START] - Start Fine-tuning Model Stage $stage of $iterative_prune_train_step"
    train_data_path="${tokenized_dataset_dir}/tokenized_dataset_stage_${stage}_of_${iterative_prune_train_step}"
    train_model_path="${pruned_model_save_dir}/adaptive_prune_stage_${stage}_of_${iterative_prune_train_step}/output_model.bin"
    output_dir="${trained_model_save_dir}/train_stage_${stage}_of_${iterative_prune_train_step}"
    echo "Training Model ${train_model_path}"
    echo "Current base steps ${base_training_steps}"
    
    accelerate launch --config_file=fsdp_config_2.yaml ../post_train.py \
        --data_path ${train_data_path} \
        --prune_model ${train_model_path} \
        --dataset_tokenized \
        --save_log_name ${save_log_name} \
        --output_dir ${output_dir} \
        --return_pth ${tmp_result_name} \
        --learning_rate ${learning_rate} \
        --min_learning_rate ${min_learning_rate} \
        --lr_scheduler ${lr_scheduler} \
        --batch_size ${batch_size} \
        --micro_batch_size ${micro_batch_size} \
        --train_epochs 1 \
        --resume_previous_stages \
        --total_training_steps ${total_training_steps} \
        --base_training_steps ${base_training_steps} \
        --total_warmup_steps ${total_warmup_steps}
    
    cur_training_steps=$(cat ${tmp_result_name})
    rm -f ${tmp_result_name}
    echo "Returned current iteration training steps: $cur_training_steps"
    base_training_steps=$(( ${base_training_steps} + ${cur_training_steps} ))
    echo "Update base training steps to: $base_training_steps"

    latest_checkpoint=$(ls -vd ${output_dir}/checkpoint-* | tail -n 1)
    if [ -z "$latest_checkpoint" ]; then
        echo "Error: No checkpoint found in ${output_dir}"
        exit 1
    fi
    model_path="${latest_checkpoint}"
    echo "Using latest checkpoint: ${model_path}"
    echo "[FINISH] - Finish Fine-tuning Model Stage $stage"

done

echo "[START] - Copying final model to output directory"
mkdir -p "${output_model_save_dir}"
cp -r "${model_path}/." "${output_model_save_dir}/"
echo "[FINISH] - Final model copied to: ${output_model_save_dir}"

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
echo "----------------------------------------"
echo "Total execution time: ${hours} hours and ${minutes} minutes"
echo "----------------------------------------"

kill $GPU_MONITOR_PID

echo "GPU usage logs have been saved to ${base_dir}/gpu_logs/gpu_usage.log"
