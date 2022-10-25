#!/bin/bash

#SBATCH --job-name=training
#SBATCH --time=06:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=25
#SBATCH --mem=64G
#SBATCH --partition=regular

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=s.ismaiel@student.rug.nl

module purge
module load Python/3.9.6-GCCcore-11.2.0

python3 --version

python3 -m venv /data/$USER/.envs/training
source /data/$USER/.envs/training/bin/activate

pip install --upgrade pip
pip install --upgrade wheel

cd /home/$USER/training

pip install -r requirements.txt

mkdir models
mkdir results
mkdir preprocessed

python3 scraper.py ${SLURM_ARRAY_TASK_ID}