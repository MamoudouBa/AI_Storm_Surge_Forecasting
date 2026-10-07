#!/bin/bash
#SBATCH --job-name=sbatch-gpu
#SBATCH --chdir=/contrib/Mamoudou.Ba/storm_surge/

echo RUNNING JOB > /contrib/Mamoudou.Ba/gru_surge_training.out


nvidia-smi

/contrib/Mamoudou.Ba/storm_surge/gru_multi_hours_predict_with_history_past_predict.py
#/contrib/Mamoudou.Ba/storm_surge/gru_surge_tuned_eager.py
#/contrib/Mamoudou.Ba/storm_surge/gru_surge_tuned.py
#/contrib/Mamoudou.Ba/lstm_multi_hours_training.py
#/contrib/Mamoudou.Ba/lstm_storm_surge_multi_steps.py
#/contrib/Mamoudou.Ba/gru_storm_surge.py
#/contrib/Mamoudou.Ba/lstm_storm_surge.py
#/contrib/Mamoudou.Ba/gru_predict_tuning.py
