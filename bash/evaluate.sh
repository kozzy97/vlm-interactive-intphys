#!/bin/bash
#SBATCH --job-name=int-eval
#SBATCH --output=../logs/interaction-proj/eval/int_eval_%A_%a.out
#SBATCH --error=../logs/interaction-proj/eval/int_eval_%A_%a.err
#SBATCH --array=1-39:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.voudouris@helmholtz-munich.de

#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --nice=10000

models=(
    "unsloth/Qwen2.5-VL-7B-Instruct"
)

n_examples=(0 3 5)

# Total combinations = number of models * number of n_examples
total_models=${#models[@]}
total_nexamples=${#n_examples[@]}

# Convert SLURM_ARRAY_TASK_ID into zero-based index
task_idx=$((SLURM_ARRAY_TASK_ID - 1))

# Determine indices for model and n_examples
model_index=$(( task_idx / total_nexamples ))
n_index=$(( task_idx % total_nexamples ))

current_model=${models[$model_index]}
current_n=${n_examples[$n_index]}
# current_model=${models[$task_idx]}

cd <path_to_your_project_root>

conda init
source ~/.bashrc
conda activate vlm_grpo

python scripts/evaluate.py $current_model --dataset_name "lsbuschoff/interact_single_static_eval" --num_in_context_examples $current_n