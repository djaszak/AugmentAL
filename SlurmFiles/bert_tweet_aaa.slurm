#!/bin/bash
#SBATCH --nodes=1 # request 2 nodes
#SBATCH --partition=alpha
#SBATCH --mincpus=1 # allocate one task per node...
#SBATCH --ntasks=1 # ...which means 2 tasks in total (see note below)
#SBATCH --cpus-per-task=32 # use 6 threads per task
#SBATCH --gres=gpu:8 # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --time=150:00:00 # run for 1 hour
#SBATCH --account=p_ml_il # account CPU time to project p_number_crunch
#SBATCH --mem=123750
#SBATCH --mail-user=dennis.jaszak@mailbox.tu-dresden.de
#SBATCH --output=/data/horse/ws/s8822750-active-learning-data-augmentation/bart/logs/out-%A_%a.txt
#SBATCH --error=/data/horse/ws/s8822750-active-learning-data-augmentation/bart/logs/error-%A_%a.txt

module load release/23.04  GCCcore/12.2.0
module load Python/3.10.8

source /data/horse/ws/s8822750-active-learning-data-augmentation/active-learning-venv/bin/activate

export HF_MODULE_CACHE="/data/horse/ws/s8822750-active-learning-data-augmentation/hf-cache"
export TRANSFORMERS_CACHE="/data/horse/ws/s8822750-active-learning-data-augmentation/hf-cache"
export HF_DATASETS_CACHE="/data/horse/ws/s8822750-active-learning-data-augmentation/hf-cache"

srun python /home/s8822750/AugmentAL/AugmentAL/bert_substitute_tweet_aaa.py
