#!/bin/bash

#SBATCH --nodes=1 # request one node

#SBATCH --cpus-per-task=7  # ask for 8 cpus

#SBATCH --mem=10G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 128 GB of ram.

#SBATCH --partition=speedy

#SBATCH --time=1-10:00:00 # ask that the job be allowed to run for 2 days, 2 hours, 30 minutes, and 2 seconds.  --time=2-02:30:02

#SBATCH --array=0-0 #specify how many times you want a job to run, we have a total of 7 array spaces

# everything below this line is optional, but are nice to have quality of life things

#SBATCH --output=imExperiment.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out

#SBATCH --error=imExperiment.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err

#SBATCH --job-name="bandits" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

# under this we just do what we would normally do to run the program, everything above this line is used by slurm to tell it what your job needs for resources

# let's load the modules we need to do what we're going to do

module load gcc/10.2.0-zuvaafu

#module load python/3.8.8-ucekvff

source /work/LAS/cjquinn-lab/Guanyu/aaai22/env/bin/activate


# let's make sure we're where we expect to be in the filesystem tree   cd /work/LAS/whatever-lab/user/thing-im-working-on

cd /work/LAS/cjquinn-lab/Guanyu/uai23/k_submodular/k_submodular/influence_maximization_is/

# the commands we're running are below

python experiment_fb.py --n-jobs 7 --B 1 2 3 4  --n-mc 500 --mode final
