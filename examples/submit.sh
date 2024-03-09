#!/bin/bash
#SBATCH --job-name=cpu256
#SBATCH --time=3-00:00:00
#SBATCH --mem=100GB
#SBATCH -p serc
#SBATCH -c 20
#SBATCH --ntasks=8
#SBATCH -o ./../sbatch_output_logs/out_test.%j.out
#SBATCH -e ./../sbatch_output_logs/err_test.%j.err

# below you run/call your code, load modules, python, Matlab, R, etc.
# and do any other scripting you want
# lines that begin with #SBATCH are directives (requests) to the scheduler-SLURM module load python/3.6.1
julia train_Elle_FNO.jl --data_path ./../data/train_data/grid256/ --model_dimension 256