#!/bin/bash
#SBATCH --job-name=mtl4ppm_all_experiments
#SBATCH --cpus-per-task=30
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lkirchdo/MTL4PPM

# Define lists of values for each argument
DATASETS=("HelpDesk.csv" "P2P.csv" "Production.csv" "Sepsis.csv" "BPIC_Challenge_2012C.csv" "BPIC15_1.csv" "BPIC20_DomesticDeclarations.csv" "BPIC20_InternationalDeclarations.csv")
TASKS=("next_activity next_time" "next_activity remaining_time" "next_time remaining_time" "multi")
MODELS=("LSTM" "CNN" "Transformer")
MTL_METHODS=("EW" "DWA" "GLS" "RLW" "UW" "UW_O" "UW_SO" "Scalarization" "IMTL" "Nash_MTL" "GradNorm" "GradDrop" "PCGrad" "CAGrad")
EPOCHS=200
LEARNING_RATES=(0.0001 0.001 0.01 0.1)
SEEDS=(42 43 44)


# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for mtl_method in "${MTL_METHODS[@]}"; do
                    for seed in "${SEEDS[@]}"; do

                        # Method-specific hyperparameters
                        case "$mtl_method" in
                            "UW_SO")
                                for temp in 1 3 5 7 10 15 20; do
                                    echo "Running: $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, $mtl_method, seed=$seed, temp=$temp"
                                    python main.py \
                                        --data_path "data/$dataset" \
                                        --tasks $task \
                                        --model "$model" \
                                        --epochs "$EPOCHS" \
                                        --learning_rate "$lr" \
                                        --seed "$seed" \
                                        --weighting "$mtl_method" \
                                        --use_softmax \
                                        --T "$temp"
                                done
                                continue
                                ;;
                            "Scalarization")
                                # Count number of tasks by splitting the space-separated string
                                IFS=' ' read -r -a task_array <<< "$task"
                                num_tasks=${#task_array[@]}

                                if [ "$num_tasks" -eq 2 ]; then
                                    weight_sets=("0.1 0.9" "0.2 0.8" "0.3 0.7" "0.4 0.6" "0.5 0.5" "0.6 0.4" "0.7 0.3" "0.8 0.2" "0.9 0.1")
                                elif [ "$num_tasks" -eq 3 ]; then
                                    weight_sets=(
                                    "0.1 0.1 0.8"
                                    "0.1 0.2 0.7"
                                    "0.1 0.3 0.6"
                                    "0.1 0.4 0.5"
                                    "0.1 0.5 0.4"
                                    "0.1 0.6 0.3"
                                    "0.1 0.7 0.2"
                                    "0.1 0.8 0.1"
                                    "0.2 0.1 0.7"
                                    "0.2 0.2 0.6"
                                    "0.2 0.3 0.5"
                                    "0.2 0.4 0.4"
                                    "0.2 0.5 0.3"
                                    "0.2 0.6 0.2"
                                    "0.2 0.7 0.1"
                                    "0.3 0.1 0.6"
                                    "0.3 0.2 0.5"
                                    "0.3 0.3 0.4"
                                    "0.3 0.4 0.3"
                                    "0.3 0.5 0.2"
                                    "0.3 0.6 0.1"
                                    "0.4 0.1 0.5"
                                    "0.4 0.2 0.4"
                                    "0.4 0.3 0.3"
                                    "0.4 0.4 0.2"
                                    "0.4 0.5 0.1"
                                    "0.5 0.1 0.4"
                                    "0.5 0.2 0.3"
                                    "0.5 0.3 0.2"
                                    "0.5 0.4 0.1"
                                    "0.6 0.1 0.3"
                                    "0.6 0.2 0.2"
                                    "0.6 0.3 0.1"
                                    "0.7 0.1 0.2"
                                    "0.7 0.2 0.1"
                                    "0.8 0.1 0.1"
                                    )
                                else
                                    echo "Unsupported number of tasks ($num_tasks) for Scalarization"
                                    continue
                                fi

                                for weights in "${weight_sets[@]}"; do
                                    echo "Running: $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, $mtl_method, seed=$seed, scalar_weights=$weights"
                                    python main.py \
                                        --data_path "data/$dataset" \
                                        --tasks $task \
                                        --model "$model" \
                                        --epochs "$EPOCHS" \
                                        --learning_rate "$lr" \
                                        --seed "$seed" \
                                        --weighting "$mtl_method" \
                                        --scalar_weights $weights
                                done
                                continue
                                ;;
                            "GradNorm")
                                for alpha in 0.5 1.0 1.5 2.0; do # following the setup of Xin et al.
                                    echo "Running: $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, $mtl_method, seed=$seed, alpha=$alpha"
                                    python main.py \
                                        --data_path "data/$dataset" \
                                        --tasks $task \
                                        --model "$model" \
                                        --epochs "$EPOCHS" \
                                        --learning_rate "$lr" \
                                        --seed "$seed" \
                                        --weighting "$mtl_method" \
                                        --alpha "$alpha"
                                done
                                continue
                                ;;
                            "CAGrad")
                                for calpha in 0.1 0.3 0.5 0.7 0.9; do # marginally smaller grid than in the original paper
                                    echo "Running: $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, $mtl_method, seed=$seed, calpha=$calpha"
                                    python main.py \
                                        --data_path "data/$dataset" \
                                        --tasks $task \
                                        --model "$model" \
                                        --epochs "$EPOCHS" \
                                        --learning_rate "$lr" \
                                        --seed "$seed" \
                                        --weighting "$mtl_method" \
                                        --calpha "$calpha"
                                done
                                continue
                                ;;
                            "GradDrop") # need to set rep_grad=True
                                echo "Running: $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, $mtl_method, seed=$seed"
                                python main.py \
                                    --data_path "data/$dataset" \
                                    --tasks $task \
                                    --model "$model" \
                                    --epochs "$EPOCHS" \
                                    --learning_rate "$lr" \
                                    --seed "$seed" \
                                    --weighting "$mtl_method" \
                                    --rep_grad
                                continue
                                ;;
                            *) # no hyperparameters to tune for other methods
                                echo "Running: $dataset, $task, $model, epochs=$EPOCHS, lr=$lr, $mtl_method, seed=$seed"
                                python main.py \
                                    --data_path "data/$dataset" \
                                    --tasks $task \
                                    --model "$model" \
                                    --epochs "$EPOCHS" \
                                    --learning_rate "$lr" \
                                    --seed "$seed" \
                                    --weighting "$mtl_method"
                                continue
                                ;;
                        esac
                    done
                done
            done
        done
    done
done
