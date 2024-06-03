#!/bin/bash
#SBATCH --nodes=1 # request 2 nodes
#SBATCH --partition=alpha
#SBATCH --mincpus=1 # allocate one task per node...
#SBATCH --ntasks=1 # ...which means 2 tasks in total (see note below)
#SBATCH --cpus-per-task=12 # use 6 threads per task
#SBATCH --gres=gpu:1 # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --time=60:00:00 # run for 1 hour
#SBATCH --account=p_ml_il # account CPU time to project p_number_crunch
#SBATCH --mem=990000
#SBATCH --mail-user=dennis.jaszak@mailbox.tu-dresden.de
#SBATCH --output=/beegfs/ws/0/s8822750-active-learning-data-augmentation/no_augmentation/logs/out-%A_%a.txt
#SBATCH --error=/beegfs/ws/0/s8822750-active-learning-data-augmentation/no_augmentation/logs/error-%A_%a.txt

module load release/23.04  GCCcore/12.2.0
module load Python/3.10.8

source /beegfs/ws/0/s8822750-active-learning-data-augmentation/active-learning-venv/bin/activate

srun python /home/s8822750/AugmentAL/AugmentAL/no_augmentation_random_imdb.py
