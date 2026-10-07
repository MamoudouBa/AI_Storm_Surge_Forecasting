#!/bin/bash
#SBATCH --job-name=sbatch-gpu
#SBATCH --chdir=/contrib/Mamoudou.Ba/storm_surge/

echo RUNNING JOB > /contrib/Mamoudou.Ba/gru_surge_training.out


nvidia-smi

/contrib/Mamoudou.Ba/storm_surge/gru_multi_hours_training_optimaze_hp_with_sequences_grouping.py
#/contrib/Mamoudou.Ba/storm_surge/gru_forecast_gfs_tuned_grouped.py
