#!/bin/bash
#SBATCH --job-name=interact
#SBATCH --output=logs/grpo_%A.out
#SBATCH --error=logs/grpo_%A.err

#SBATCH -p gpu_p
#SBATCH --qos=gpu_long
#SBATCH --cpus-per-task=28
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

# Activate conda properly
source <path_to_conda.sh>
conda activate vlm_grpo

# Run the script
python scripts/grpo.py --model_name "unsloth/Qwen2.5-VL-7B-Instruct" # --triplet