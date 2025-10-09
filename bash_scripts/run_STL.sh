#!/bin/bash
#SBATCH --job-name=mtl4ppm_STL
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lkirchdo/MTL4PPM

# Ensure conda is available
source ~/.bashrc || { echo "Failed to source ~/.bashrc"; exit 1; }

# Activate conda environment
conda activate MTL4PPM || { echo "Failed to activate MTL4PPM environment"; exit 1; }

# Navigate to project directory
echo "Changing directory to the MTL4PPM repository..."
cd /ceph/lkirchdo/MTL4PPM || { echo "Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

# -------------------------------
echo "Starting Python script execution..."
echo "-------------------------------------"


# DATASETS=("HelpDesk.csv" "P2P.csv" "Production.csv" "Sepsis.csv" "BPI_Challenge_2012C.csv" "BPIC15_1.csv" "BPIC20_DomesticDeclarations.csv" "BPIC20_InternationalDeclarations.csv" "BPI_Challenge_2013_incidents.csv")
DATASETS=("BPI_Challenge_2013_incidents.csv")
TASKS=("next_activity" "next_time" "remaining_time")
MODELS=("CNN" "LSTM" "Transformer")
EPOCHS=200
LEARNING_RATES=(0.0001 0.001 0.01)
SEEDS=(42 123 2025)


# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    echo "Running experiment with $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, seed=$seed"
                    python main.py \
                    --data_path "data/$dataset" \
                    --tasks "$task" \
                    --model "$model" \
                    --epochs "$EPOCHS" \
                    --learning_rate "$lr" \
                    --seed "$seed"
                done
            done
        done
    done
done