#!/bin/bash
#SBATCH --job-name=py-matinv     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=mz1231@princeton.edu

module purge
module load anaconda3/2024.2
conda activate cosjp

export TORCH_HOME=/scratch/network/mz1231/pretrained

torchrun --nproc_per_node 1 /scratch/network/mz1231/image_corruption/HCL/custom_test.py evaluate -c /scratch/network/mz1231/image_corruption/HCL/configs/custom_eval.yml