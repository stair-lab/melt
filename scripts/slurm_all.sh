#!/bin/bash
# Job name:
#SBATCH --job-name=ura-llama
#
# Partition:
#SBATCH --partition=gpu
#
# Number of nodes:
#SBATCH --nodes=10
#
# Number of tasks across all nodes:
#SBATCH --ntasks=92
#
# Number of gpus per node:
#SBATCH --gres=gpu:92
#SBATCH -C GPU_MEM:24GB
#
# Number of gpus per task:
#SBATCH --gpus-per-task=1
#
# Number of CPUs per task:
# Always at least twice the number of GPUs per task
#SBATCH --cpus-per-task=8
#
# Memory per CPU:
#SBATCH --mem-per-cpu=2G
#
## Command(s) to run:

srun -N 1 -n 18 --gpus 18 --exclusive scripts/question_answering.sh &
srun -N 1 -n 18 --gpus 18 --exclusive scripts/ummarization.sh &
srun -N 1 -n 8 --gpus 8 --exclusive scripts/translation.sh &
srun -N 1 -n 6 --gpus 6 --exclusive scripts/knowledge.sh &
srun -N 1 -n 6 --gpus 6 --exclusive scripts/information_retrieval.sh &
srun -N 1 -n 6 --gpus 6 --exclusive scripts/sentiment_analysis.sh &
srun -N 1 -n 6 --gpus 6 --exclusive scripts/text_classification.sh &
srun -N 1 -n 6 --gpus 6 --exclusive scripts/toxic_detection.sh &
srun -N 1 -n 6 --gpus 6 --exclusive scripts/language modelling.sh &
srun -N 1 -n 12 --gpus 12 --exclusive scripts/reasoning,sh &
wait
