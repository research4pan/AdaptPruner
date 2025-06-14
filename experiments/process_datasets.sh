#!/bin/bash

model="$1"
save_dir="$2"
echo "${model}"
repeat_dataset=1
seperate_strategy=linear
cutoff_len=1024
data_path='Open-Orca/OpenOrca,teknium/OpenHermes-2.5,databricks/databricks-dolly-15k,allenai/WildChat-1M,lmsys/lmsys-chat-1m,HuggingFaceH4/ultrachat_200k,openbmb/UltraInteract_sft,yahma/alpaca-cleaned,O1-OPEN/OpenO1-SFT'



case "$model" in
    "MobileLLM-350M")
        model_path='facebook/MobileLLM-350M'
        model_name='MobileLLM-350M'
        iterative_prune_train_step=20
        ;;
    "MobileLLM-600M")
        model_path='facebook/MobileLLM-600M'
        model_name='MobileLLM-600M'
        iterative_prune_train_step=9
        ;;
    "MobileLLM-1B")
        model_path='facebook/MobileLLM-1B'
        model_name='MobileLLM-1B'
        iterative_prune_train_step=10
        ;;
    "Qwen2.50.5B")
        model_path='Qwen/Qwen2.5-0.5B'
        model_name='Qwen2.50.5B'
        iterative_prune_train_step=5
        ;;
    "Deepseek-R1-Distill-Qwen-1.5B")
        model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        model_name='Deepseek_1.5B'
        iterative_prune_train_step=11
        ;;
    *)
        echo "model name wrong"
        ;;
esac



base_dir="${save_dir}/processed_datasets_${model_name}_${iterative_prune_train_step}_steps_repeat_${repeat_dataset}_strategy_${seperate_strategy}_instruct_1"
save_log_name="${base_dir}/log"
save_log_pth="${save_log_name}/log.out"
tmp_result_name="${base_dir}/tmp_result.out"
tokenized_dataset_dir="${base_dir}/tokenized_dataset"

mkdir -p "$(dirname "${save_log_pth}")"
> "${save_log_pth}"
exec &> >(tee -a ${save_log_pth})

echo "Processing data for model: ${model_name}"
echo "Logging path: ${save_log_pth}"
echo "Temporary returned value path: ${tmp_result_name}"
echo "Model path: ${model_path}"
echo "Dataset path: ${data_path}"
echo "Tokenized dataset path: ${tokenized_dataset_dir}"
echo "Iterative prune and train steps: ${iterative_prune_train_step}"

echo "[START] - Start Processing dataset"
echo "Cutoff length: ${cutoff_len}"
echo "Dataset repeat: ${repeat_dataset}"
echo "Seperate strategy: ${seperate_strategy}"

python ../utils/process_dataset.py \
    --model ${model_path} \
    --data_path ${data_path} \
    --output_dir ${tokenized_dataset_dir} \
    --return_pth ${tmp_result_name} \
    --save_log_name ${save_log_name} \
    --cutoff_len ${cutoff_len} \
    --iterative_prune_train_step ${iterative_prune_train_step} \
    --repeat_dataset ${repeat_dataset} \
    --fraction 1.0 \
    --seperate_strategy ${seperate_strategy}

num_samples=$(cat ${tmp_result_name})

echo "Returned total sample number: $num_samples"
echo "[FINISH] - Finish Processing dataset"
