import argparse
import gym
import numpy as np
import os
import torch
import json
import time
import random

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
# from utils.wandb import init_wandb
from agents.online_agent import OnlineAgent as Agent

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 9.0,  },
    'hopper-medium-v2':              {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 9.0,  },
    'walker2d-medium-v2':            {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 1.0,  },
    'halfcheetah-medium-replay-v2':  {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 2.0,  },
    'hopper-medium-replay-v2':       {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 4.0,  },
    'walker2d-medium-replay-v2':     {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 4.0,  },
    'halfcheetah-medium-expert-v2':  {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 7.0,  },
    'hopper-medium-expert-v2':       {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 5.0,  },
    'walker2d-medium-expert-v2':     {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 5.0,  },
    'antmaze-umaze-v0':              {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50,  'gn': 2.0,  },
    'antmaze-umaze-diverse-v0':      {'lr': 1e-5,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50,  'gn': 3.0,  },
    'antmaze-medium-play-v0':        {'lr': 1e-5,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50,  'gn': 2.0,  },
    'antmaze-medium-diverse-v0':     {'lr': 1e-5,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50,  'gn': 1.0,  },
    'antmaze-large-play-v0':         {'lr': 1e-5,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50,  'gn': 10.0, },
    'antmaze-large-diverse-v0':      {'lr': 1e-5,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50,  'gn': 7.0,  },
    'pen-human-v1':                  {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50,  'gn': 7.0,  },
    'pen-cloned-v1':                 {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50,  'gn': 8.0,  },
    'kitchen-complete-v0':           {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 9.0,  },
    'kitchen-partial-v0':            {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 10.0, },
    'kitchen-mixed-v0':              {'lr': 1e-5,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50,  'gn': 10.0, },
}

def make_env(args):
    def _thunk():
        env = gym.make(args.env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.seed(seed + rank)
        # env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
        # env = gym.wrappers.NormalizeActionWrapper(env)
        return env
    return _thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def train_agent(state_dim, action_dim, max_action, device, output_dir, writer, args):

    agent = Agent(state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                model=args.model,
                device=device,
                discount=args.discount,
                tau=args.tau,
                max_q_backup=args.max_q_backup,
                beta_schedule=args.beta_schedule,
                n_timesteps=args.T,
                lr=args.lr,
                lr_decay=args.lr_decay,
                grad_norm=args.gn)

    if args.load_model != "":
        agent.load_model(args.load_model, args.load_id)
        print(f"Loaded agent from: {args.load_model} with id: {args.load_id}")

    envs = gym.vector.SyncVectorEnv([make_env(args) for _ in range(args.num_envs)])

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        args.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    evaluations = []

    obs = envs.reset()
    for global_step in range(args.total_timesteps): 
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)     
        obs = obs.reshape(args.num_envs, -1) 
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = []
            for o in obs:   
                actions.append(agent.sample_action(np.array(o)))
            actions = np.array(actions)

        next_obs, rewards, dones, infos = envs.step(actions)

        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        real_next_obs = real_next_obs.reshape(args.num_envs, -1) 

        rb.add(obs.astype(np.float32), real_next_obs.astype(np.float32), actions, rewards, dones, infos)
        obs = next_obs

        # train
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                loss_metric = agent.train(rb,
                                        iterations=1,
                                        batch_size=args.batch_size,
                                        log_writer=writer)
                curr_time = time.time()
                actor_loss = np.mean(loss_metric['actor_loss'])
                critic_loss = np.mean(loss_metric['critic_loss'])
                used_time = curr_time - start_time

        if global_step > args.learning_starts and global_step % args.eval_frequency == 0:
            # Evaluation
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                                eval_episodes=args.eval_episodes)

            evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                                 global_step])
            np.save(os.path.join(output_dir, "eval"), evaluations)
            utils.print_banner(f"Train step: {global_step}", separator="*", num_star=90)
            print("SPS:", int(global_step / (time.time() - start_time)))
            
            # Logging
            if actor_loss is not None and critic_loss is not None:
                logger.record_tabular('Trained Steps', global_step)
                logger.record_tabular('Actor Loss', actor_loss)
                logger.record_tabular('Critic Loss', critic_loss)
                logger.record_tabular('Time', used_time)

                writer.add_scalar(f"charts/time", used_time, global_step)

            logger.record_tabular('Average Episodic Reward', eval_res)
            logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
            logger.dump_tabular()

            writer.add_scalar(f"eval_charts/eval_reward", eval_res, global_step)
            writer.add_scalar(f"eval_charts/eval_reward_std", eval_res_std, global_step)
            writer.add_scalar(f"eval_charts/eval_norm_reward", eval_norm_res, global_step)
            writer.add_scalar(f"eval_charts/eval_norm_reward_std", eval_norm_res_std, global_step)

        if args.save_best_model:
            agent.save_model(output_dir, global_step)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    policy.model.eval()
    policy.actor.eval()

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            # eval_env.render()
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)
    policy.model.train()
    policy.actor.train()
    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_envs", default=2, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument('--wandb_activate', type=bool, default=False, help='activate wandb for logging')
    parser.add_argument('--wandb_entity', type=str, default='', help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
    parser.add_argument('--wandb_name', type=str, default='', help='wandb name')

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    
    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--buffer_size", type=int, default=100000, help="the replay memory buffer size")
    parser.add_argument("--learning_starts", type=int, default=0, help="timestep to start learning")
    parser.add_argument("--train_frequency", type=int, default=4, help="the frequency of training")
    parser.add_argument("--eval_frequency", type=int, default=10000, help="the frequency of training")
    parser.add_argument("--start_e", type=float, default=1, help="the starting epsilon for exploration")
    parser.add_argument("--end_e", type=float, default=0.01, help="the ending epsilon for exploration")
    parser.add_argument("--exploration_fraction", type=float, default=0.10, help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    
    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)

    ### Algo Choice ###
    parser.add_argument("--algo", default="online", type=str) 
    parser.add_argument("--model", default="consistency", type=str)  # ['diffusion', 'consistency']
    parser.add_argument("--eta", default=-1.0, type=float)
    parser.add_argument('--load_model', type=str, default='', help='load pretrained checkpoint')
    parser.add_argument('--load_id', type=str, default='', help='model id to load')

    ### Pre-specified ###
    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--max_q_backup", action='store_true')
    # parser.add_argument("--reward_tune", default='no', type=str)
    # parser.add_argument("--gn", default=-1.0, type=float)

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']

    # if args.wandb_activate:
    #     args.wandb_project = 'consistency-rl-online'
    #     args.wandb_group = f'{args.exp}'
    #     args.wandb_name = f'{args.env_name}_{args.algo}_{args.model}_{args.exp}'
    #     init_wandb(args)

    writer = SummaryWriter(f"runs/{args.env_name}_{args.algo}_{args.model}_{args.exp}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|{args.model}-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'

    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    if os.path.exists(os.path.join(results_dir, 'variant.json')):
        raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"{args.model}-policies-RL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                writer,
                args)
