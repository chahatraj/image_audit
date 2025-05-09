#!/bin/bash

#SBATCH --job-name=sd690
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=90GB
#SBATCH --time=2-23:59:58
#SBATCH --array=2-2
#SBATCH --output=/scratch/amukher6/logs/image_audit/out/sd690_n.%A_%a.out.txt
#SBATCH --error=/scratch/amukher6/logs/image_audit/err/sd690_n.%A_%a.err.txt

#nvidia-smi

# Read arguments from the config file
readarray -t ARGS < splits.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS
python gen_image.py $ARGUMENTS
