#!/bin/bash -l
#SBATCH --job-name=a_finetune00
#SBATCH --cpus-per-task=24
#SBATCH --output=output_finetune00.txt
#SBATCH --gres=gpu:3
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=2-00:00:00
#SBATCH --exclude=c[1-16]
# #SBATCH --exclude=c[1-18,20]
# #SBATCH --exclude=c[11,17,18,19,20]
# #SBATCH --exclude=c[11]
# #SBATCH --nodelist=c[9,10,12,13,17,18,19,20]
# #SBATCH --exclude=c[1-16]

module --silent anaconda3/2023.09-0-gcc-13.2.0-6gotrib
conda activate gofaster00

echo "GPU CHECK"
nvidia-smi

echo "==============================================================================================================="
echo "PYTHON SCRIPT:"

accelerate launch finetune.py

echo "python path executed"

echo "SBATCH FINISHED"
