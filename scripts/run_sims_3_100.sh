#!/bin/bash
#SBATCH --account=p31628 ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=short ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --time=4:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes=1 ## how many computers/nodes do you need (no default)
#SBATCH --ntasks-per-node=1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=sims3-100 ## When you run squeue -u 
#SBATCH --output=outfiles/%x-%j.out

eval "$(conda shell.bash hook)"
conda activate derm_sims
python3 simulations_parallel.py --scenario low_variability more_aggressive more_conservative --derms 100