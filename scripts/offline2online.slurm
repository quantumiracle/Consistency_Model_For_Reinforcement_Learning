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


echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

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

seed='1'


# # consistency
# declare -a paths=(
#                 "paper_results/consist_ql/halfcheetah-medium-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
#                 "paper_results/consist_ql/hopper-medium-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 "paper_results/consist_ql/walker2d-medium-v2|consistency${seed}|consistency-ql|T-5|lr_decay|ms-offline|k-1|${seed}"
#                 "paper_results/consist_ql/halfcheetah-medium-replay-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-0|${seed}"
#                 "paper_results/consist_ql/hopper-medium-replay-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 "paper_results/consist_ql/walker2d-medium-replay-v2|consistency${seed}|consistency-ql|T-5|lr_decay|ms-offline|k-10|${seed}"
#                 "paper_results/consist_ql/halfcheetah-medium-expert-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-0|${seed}"
#                 "paper_results/consist_ql/hopper-medium-expert-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 "paper_results/consist_ql/walker2d-medium-expert-v2|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
#                 # "paper_results/consist_ql/antmaze-umaze-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/antmaze-umaze-diverse-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/antmaze-medium-play-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
#                 # "paper_results/consist_ql/antmaze-medium-diverse-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
#                 # "paper_results/consist_ql/antmaze-large-play-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/antmaze-large-diverse-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
#                 # "paper_results/consist_ql/pen-human-v1|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/pen-cloned-v1|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/kitchen-complete-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/kitchen-partial-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
#                 # "paper_results/consist_ql/kitchen-mixed-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-0|${seed}"
# )

# declare -a models=('consistency')  # consistency, diffusion

# mkdir -p log/$DATE

# for i in ${!envs[@]}; do
#     for j in ${!models[@]}; do
#         echo python online.py --env_name ${envs[$i]} --model ${models[$j]} --exp consistency_online_load_offline${seed} --num_envs 3 --load_id 'offline' --load_model ${paths[$i]} --device $((i % 8))  output log to: log/$DATE/${envs[$i]}_${models[$j]}.log &
#         nohup python -W ignore online.py --env_name ${envs[$i]} --model ${models[$j]} --exp consistency_online_load_offline${seed}  --num_envs 3 --load_id 'offline' --load_model ${paths[$i]} --device $((i % 8)) >> log/$DATE/${envs[$i]}_${models[$j]}.log &        
#     done
# done
# wait

# diffusion
declare -a paths=(
                "results/halfcheetah-medium-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0"
                "results/hopper-medium-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-2|0"
                "results/walker2d-medium-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0"
                "results/halfcheetah-medium-replay-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-0|0"
                "results/hopper-medium-replay-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-2|0"
                "results/walker2d-medium-replay-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0"
                "results/halfcheetah-medium-expert-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-0|0"
                "results/hopper-medium-expert-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-2|0"
                "results/walker2d-medium-expert-v2|test|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0"
                # "paper_results/consist_ql/antmaze-umaze-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/antmaze-umaze-diverse-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/antmaze-medium-play-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
                # "paper_results/consist_ql/antmaze-medium-diverse-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
                # "paper_results/consist_ql/antmaze-large-play-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/antmaze-large-diverse-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-1|${seed}"
                # "paper_results/consist_ql/pen-human-v1|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/pen-cloned-v1|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/kitchen-complete-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/kitchen-partial-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-2|${seed}"
                # "paper_results/consist_ql/kitchen-mixed-v0|consistency${seed}|consistency-ql|T-2|lr_decay|ms-offline|k-0|${seed}"
)

declare -a models=('diffusion')  # consistency, diffusion

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!models[@]}; do
        echo python online.py --env_name ${envs[$i]} --model ${models[$j]} --exp diffusion_online_load_online${seed} --num_envs 3 --load_id 'online' --load_model ${paths[$i]} --device $((i % 8))  output log to: log/$DATE/${envs[$i]}_${models[$j]}.log &
        nohup python -W ignore online.py --env_name ${envs[$i]} --model ${models[$j]} --exp diffusion_online_load_online${seed}  --num_envs 3 --load_id 'online' --load_model ${paths[$i]} --device $((i % 8)) >> log/$DATE/${envs[$i]}_${models[$j]}.log &        
    done
done
wait