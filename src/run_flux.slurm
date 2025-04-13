#!/bin/bash

#SBATCH --job-name=flux690
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=90GB
#SBATCH --time=2-23:59:58
#SBATCH --array=1-1
#SBATCH --output=/scratch/craj/logs/vqa/out/flux690_n.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/vqa/err/flux690_n.%A_%a.err.txt

#nvidia-smi

# Read arguments from the config file
readarray -t ARGS < splits.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS
python gen_image.py $ARGUMENTS
