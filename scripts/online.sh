echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
declare -a envs=('halfcheetah-medium-v2'
                'hopper-medium-v2'
                'walker2d-medium-v2'
                'halfcheetah-medium-replay-v2'
                'hopper-medium-replay-v2'
                'walker2d-medium-replay-v2'
                'halfcheetah-medium-expert-v2'
                'hopper-medium-expert-v2'
                'walker2d-medium-expert-v2'
                'antmaze-umaze-v0'
                'antmaze-umaze-diverse-v0'
                'antmaze-medium-play-v0'
                'antmaze-medium-diverse-v0'
                'antmaze-large-play-v0'
                'antmaze-large-diverse-v0'
                'pen-human-v1'
                'pen-cloned-v1'
                'kitchen-complete-v0'
                'kitchen-partial-v0'
                'kitchen-mixed-v0'
)

declare -a models=('consistency')  # consistency, diffusion

# online RL
mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!models[@]}; do
        echo python online.py --env_name ${envs[$i]} --model ${models[$j]} --num_envs 3 --device $((i % 8))  output log to: log/$DATE/${envs[$i]}_${models[$j]}.log &
        nohup python -W ignore online.py --env_name ${envs[$i]} --model ${models[$j]} --num_envs 3 --device $((i % 8)) >> log/$DATE/${envs[$i]}_${models[$j]}.log &        
    done
done

