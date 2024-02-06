#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=sample
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6       # number of CPUs per process

## number of tasks per node
#SBATCH --ntasks-per-node=11

## GPU allocation - variant A
#SBATCH --gres=gpu:8            # number of GPUs per node (gres=gpu:N)

## time out in minutes: 1- 4320
#SBATCH --time=4300

###SBATCH --gpus-per-task=1

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
module purge

# Load what we need
module load anaconda3
# export USE_CUDA=1 USE_CUDNN=1
eval "$(conda shell.bash hook)"
# module load cuda/11.0 cudnn/v8.0.3.33-cuda.11.0
# # source ~/anaconda3/etc/profile.d/conda.sh
# source ~/.bashrc
source activate x


### Section 3:
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
### This is going to run ntasks-per-node x nodes tasks with each
### task seeing all the GPUs on each node. However I am using
### the wrapper.sh example I showed before so that each task only
### sees one GPU
# srun --label train.sh


declare -a envs=(
                'halfcheetah-medium-v2'
                'hopper-medium-v2'
                'walker2d-medium-v2'
                'halfcheetah-medium-replay-v2'
                'hopper-medium-replay-v2'
                'walker2d-medium-replay-v2'
                'halfcheetah-medium-expert-v2'
                'hopper-medium-expert-v2'
                'walker2d-medium-expert-v2'
                # 'antmaze-umaze-v0'
                # 'antmaze-umaze-diverse-v0'
                # 'antmaze-medium-play-v0'
                # 'antmaze-medium-diverse-v0'
                # 'antmaze-large-play-v0'
                # 'antmaze-large-diverse-v0'
                # 'pen-human-v1'
                # 'pen-cloned-v1'
                # 'kitchen-complete-v0'
                # 'kitchen-partial-v0'
                # 'kitchen-mixed-v0'
)


declare -a methods=('offline')  # offline or online model selection
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE


declare -a seeds=($(seq 1 1 5))


# for a 8-gpus server
for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        for k in ${!seeds[@]}; do
            echo python offline.py --env_name ${envs[$i]} --device $((i % 8)) --model consistency --exp consistency_improved${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay output log to: log/$DATE/${envs[$i]}_${methods[$j]}_seed${seeds[$k]}.log &
            python -W ignore  offline.py --env_name ${envs[$i]} --device $((i % 8)) --model consistency --exp consistency_improved${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay  >> log/$DATE/${envs[$i]}_${methods[$j]}_seed${seeds[$k]}.log &        
        done
    done
done
wait
