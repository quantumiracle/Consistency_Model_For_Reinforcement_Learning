# Consistency Models for RL &mdash; Official PyTorch Implementation
Official implementation for:

**Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning**<br>
Zihan Ding, Chi Jin <br>
[https://arxiv.org/abs/2309.16984](https://arxiv.org/abs/2309.16984) <br>

## Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. Please see the ``requirements.txt`` for environment set up details.
```
pip install -r requirements.txt
```

## Run
You can use either diffusion model or consistency model.

### Dataset
First download D4RL dataset with:
```
python download_data.py
```
The data will be saved in `./dataset/`.

### Offline RL
```
# train offline RL Consistency-AC for hopper-medium-v2 task
python main.py --env_name hopper-medium-v2 --model consistency --ms offline --exp RUN_NAME --save_best_model --lr_decay
# train offline RL Diffusion-QL for walker2d-medium-expert-v2 task
python main.py --env_name walker2d-medium-expert-v2 --model diffusion --ms offline --exp RUN_NAME --save_best_model --lr_decay
```
### Online RL
From scratch:
```
# train online RL Consistency-AC for hopper-medium-v2 task
python online.py --env_name hopper-medium-v2 --num_envs 3 --model consistency --exp RUN_NAME
# train online RL Diffusion-QL for walker2d-medium-expert-v2 task
python online.py --env_name walker2d-medium-expert-v2 --num_envs 3 --model diffusion --exp RUN_NAME
```
Online RL initialized with offline pre-trained models (offline-to-online):
```
python online.py --env_name kitchen-mixed-v0 --num_envs 3 --model consistency --exp online_test --load_model 'results/path' --load_id 'online'
```
As an example, with a model saved in path `results/path/actor_online.pth`, it will be loaded for initializing the online training with the above command.

### Training Scripts
Use bash scripts:
```
bash scripts/offline.sh
bash scripts/online.sh
```

Use Slurm scripts:
```
sbatch scripts/offline.slurm
sbatch scripts/online.slurm
sbatch scripts/offline2online.slurm
```


## Citation

If you find this open source release useful, please cite in your paper:
```
@article{ding2023consistency,
  title={Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning},
  author={Ding, Zihan and Jin, Chi},
  journal={arXiv preprint arXiv:2309.16984},
  year={2023}
}
```

## Acknowledgement
We acknowledge the original official repo of [Diffusion Policy
](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)
 and corresponding paper: [https://arxiv.org/abs/2208.06193](https://arxiv.org/abs/2208.06193).
