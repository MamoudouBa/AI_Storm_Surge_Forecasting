#!/bin/bash
#SBATCH --job-name=sbatch-gpu
#SBATCH --chdir=/contrib/Mamoudou.Ba/

echo RUNNING JOB > /contrib/Mamoudou.Ba/storm_surge_data_retrieval.out


nvidia-smi

/contrib/Mamoudou.Ba/storm_surge_data_retrieval.py
