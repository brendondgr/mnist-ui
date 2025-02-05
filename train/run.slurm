#!/bin/bash
#SBATCH --job-name=MNIST
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --partition=gpu2
#SBATCH --account=gpu2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --output=/work/bdgr/MNIST/output/%j.out

LD_LIBRARY_PATH="/home/bdgr/.conda/envs/bdgr_torch/lib:$LD_LIBRARY_PATH"
export PATH="/work/bdgr/MNIST:$PATH"

python /work/bdgr/MNIST/gpu_test.py
python /work/bdgr/MNIST/train.py --epochs 100 --batch_size 256 --validate_iter 5 --save_model True --network_name MediumModel