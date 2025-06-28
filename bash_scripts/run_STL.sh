#!/bin/bash
#SBATCH --job-name=mtl4ppm_STL_experiments
#SBATCH --cpus-per-task=30
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lkirchdo/MTL4PPM


DATASETS=("HelpDesk.csv" "P2P.csv" "Production.csv" "Sepsis.csv" "BPIC_Challenge_2012C.csv" "BPIC15_1.csv" "BPIC20_DomesticDeclarations.csv" "BPIC20_InternationalDeclarations.csv")
TASKS=("next_activity" "next_time" "remaining_time")
MODELS=("LSTM" "CNN" "Transformer")
EPOCHS=200
LEARNING_RATES=(0.0001 0.001 0.01 0.1)
SEEDS=(42 43 44)


# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    echo "Running experiment with $dataset, $task, $model, epochs=$epochs, lr=$lr, seed=$seed"
                    python main.py \
                    --data_path "data/$dataset" \
                    --tasks "$task" \
                    --model "$model" \
                    --epochs "$epochs" \
                    --learning_rate "$lr" \
                    --seed "$seed"
                done
            done
        done
    done
done