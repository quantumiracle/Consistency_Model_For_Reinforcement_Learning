import gym
import numpy as np
import collections
import pickle
import d4rl


for env_name in ["halfcheetah", "hopper", "walker2d"]:
    for dataset_type in ["medium", "medium-expert", "medium-replay", "expert"]:
        name = f"{env_name}-{dataset_type}-v2"
        env = gym.make(name)
        dataset = d4rl.qlearning_dataset(env)

        with open(f"dataset/{name}.pkl", "wb") as f:
            pickle.dump(dataset, f)

other_envs = [
                'antmaze-umaze-v0',
                'antmaze-umaze-diverse-v0',
                'antmaze-medium-play-v0',
                'antmaze-medium-diverse-v0',
                'antmaze-large-play-v0',
                'antmaze-large-diverse-v0',
                'pen-human-v1',
                'pen-cloned-v1',
                'kitchen-complete-v0',
                'kitchen-partial-v0',
                'kitchen-mixed-v0',
]

for env_name in other_envs:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    with open(f"dataset/{env_name}.pkl", "wb") as f:
        pickle.dump(dataset, f)
