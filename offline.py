import argparse
import gym
import numpy as np
import os
import torch
import json
import time
import shutil
import pickle

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
# from utils.wandb import init_wandb
from torch.utils.tensorboard import SummaryWriter

diffusion_hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'T': 5,   'q_norm': False,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'T': 5,   'q_norm': False,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'T': 5,   'q_norm': False,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'T': 5,   'q_norm': False,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'T': 5,   'q_norm': False,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'T': 5,   'q_norm': False,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}

consistency_hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'T': 2,   'q_norm': False, 'scale_consis_loss': False,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 0.1,   'T': 2,  'q_norm': False, 'scale_consis_loss': False,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'T': 5,  'q_norm': True,  'scale_consis_loss': False,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 3},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'T': 2,  'q_norm': False,  'scale_consis_loss': False,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 2},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 0.1,   'T': 2,  'q_norm': False,  'scale_consis_loss': False,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 5},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 0.1,   'T': 5,  'q_norm': False,  'scale_consis_loss': False,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'T': 2,  'q_norm': False, 'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 4},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'T': 2,  'q_norm': False,  'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 5},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'T': 2,  'q_norm': True, 'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.01,   'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 4},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 0.01,   'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 4},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 0.01,   'T': 2, 'q_norm': False,  'scale_consis_loss': True,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 8},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 0.01,   'T': 2,  'q_norm': True, 'scale_consis_loss': True,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 5},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.01,  'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.01,   'T': 2,  'q_norm': True, 'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 500, 'gn': 8.0,  'top_k': 5},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.5, 'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1500 , 'gn': 2.0,  'top_k': 4},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.5, 'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1500, 'gn': 2.0, 'top_k': 1},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.5, 'T': 2,  'q_norm': True,  'scale_consis_loss': True,  'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1500, 'gn': 2.0, 'top_k': 2},
}


def train_agent(env, env_name, state_dim, action_dim, max_action, device, output_dir, writer, args):
    # Load buffer
    # dataset = d4rl.qlearning_dataset(env)
    with open(f'dataset/{env_name}.pkl', 'rb') as f:
        dataset = pickle.load(f)
              
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    if args.model  == 'diffusion':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            eta=args.eta,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
            )
    else:
        from agents.ql_consistency import Consistency_QL as Agent
        agent = Agent(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=device,
                    discount=args.discount,
                    tau=args.tau,
                    max_q_backup=args.max_q_backup,
                    n_timesteps=args.T,
                    eta=args.eta,
                    lr=args.lr,
                    lr_decay=args.lr_decay,
                    lr_maxt=args.num_epochs,
                    grad_norm=args.gn,
                    q_norm=args.q_norm,
                    steps_per_epoch=args.num_steps_per_epoch,
                    improved_CT=False,  # [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
                    )

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    start_time = time.time()
    if args.save_best_model:
        agent.save_model(output_dir, '0')
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))
        curr_time = time.time()

        bc_loss = np.mean(loss_metric['bc_loss'])
        ql_loss = np.mean(loss_metric['ql_loss'])
        actor_loss = np.mean(loss_metric['actor_loss'])
        critic_loss = np.mean(loss_metric['critic_loss'])
        used_time = curr_time - start_time

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', bc_loss)
        logger.record_tabular('QL Loss', ql_loss)
        logger.record_tabular('Actor Loss', actor_loss)
        logger.record_tabular('Critic Loss', critic_loss)
        logger.record_tabular('Time', used_time)

        writer.add_scalar(f"charts/time", used_time, training_iters)


        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes)
        bc_loss = np.mean(loss_metric['bc_loss'])
        ql_loss = np.mean(loss_metric['ql_loss'])
        actor_loss = np.mean(loss_metric['actor_loss'])
        critic_loss = np.mean(loss_metric['critic_loss'])

        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            bc_loss, ql_loss, actor_loss, critic_loss, curr_epoch])
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.dump_tabular()

        writer.add_scalar(f"eval_charts/bc_loss", bc_loss, training_iters)
        writer.add_scalar(f"eval_charts/ql_loss", ql_loss, training_iters)
        writer.add_scalar(f"eval_charts/actor_loss", actor_loss, training_iters)
        writer.add_scalar(f"eval_charts/critic_loss", critic_loss, training_iters)
        writer.add_scalar(f"eval_charts/eval_reward", eval_res, training_iters)
        writer.add_scalar(f"eval_charts/eval_reward_std", eval_res_std, training_iters)
        writer.add_scalar(f"eval_charts/eval_norm_reward", eval_norm_res, training_iters)
        writer.add_scalar(f"eval_charts/eval_norm_reward_std", eval_norm_res_std, training_iters)

        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    # online ms
    online_best_id = np.argmax(scores[:, 2])
    # offline ms
    bc_loss = scores[:, 4]
    top_k = min(len(bc_loss) - 1, args.top_k)
    offline_best_id = np.argsort(bc_loss)[top_k]

    if args.ms == 'online':
        best_res = {'model selection': args.ms, 'epoch': scores[online_best_id, -1],
                    'best normalized score avg': scores[online_best_id, 2],
                    'best normalized score std': scores[online_best_id, 3],
                    'best raw score avg': scores[online_best_id, 0],
                    'best raw score std': scores[online_best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        best_res = {'model selection': args.ms, 'epoch': scores[offline_best_id][-1],
                    'best normalized score avg': scores[offline_best_id][2],
                    'best normalized score std': scores[offline_best_id][3],
                    'best raw score avg': scores[offline_best_id][0],
                    'best raw score std': scores[offline_best_id][1]}


        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    writer.close()

    if args.save_best_model:
        # Clean up other pth files except for the selected ones
        # Create a dictionary of old and new filenames
        mapping = {
            f"actor_{int(scores[online_best_id, -1])}.pth": f"actor_online.pth",
            f"critic_{int(scores[online_best_id, -1])}.pth": f"critic_online.pth",
            f"actor_{int(scores[offline_best_id, -1])}.pth": f"actor_offline.pth",
            f"critic_{int(scores[offline_best_id, -1])}.pth": f"critic_offline.pth",
        }

        if online_best_id == offline_best_id:
            for key in mapping.keys():
                shutil.copyfile(os.path.join(output_dir, key), os.path.join(output_dir, mapping[key]))
            for file_name in os.listdir(output_dir):
                # Check if the file is in the mapping dictionary
                # only for file end with .pth
                if file_name.endswith(".pth"):
                    if file_name not in mapping.values():
                        # Delete any files not in the mapping dictionary
                        os.remove(os.path.join(output_dir, file_name))
        else:
            for file_name in os.listdir(output_dir):
                # Check if the file is in the mapping dictionary
                # only for file end with .pth
                if file_name.endswith(".pth"):
                    if file_name in mapping.keys():
                        # Rename the old file name to the new file name
                        os.rename(os.path.join(output_dir, file_name), os.path.join(output_dir, mapping[file_name]))
                    else:
                        if file_name not in mapping.values():
                            # Delete any files not in the mapping dictionary
                            os.remove(os.path.join(output_dir, file_name))



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    policy.model.eval()
    policy.actor.eval()

    scores = []
    for i in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
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
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
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

    ### Diffusion Setting ###
    parser.add_argument("--T", default=-1, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument("--model", default="diffusion", type=str)  # ['diffusion', 'consistency']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")

    parser.add_argument("--lr", default=-1., type=float)
    parser.add_argument("--eta", default=-1.0, type=float)

    parser.add_argument("--gn", default=-1.0, type=float)

    ### Pre-specified ###
    # parser.add_argument("--top_k", default=1, type=int)
    # parser.add_argument("--max_q_backup", action='store_true')
    # parser.add_argument("--reward_tune", default='no', type=str)

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    if args.model == 'diffusion':
        hyperparameters = diffusion_hyperparameters
    elif args.model == 'consistency':
        hyperparameters = consistency_hyperparameters
        args.scale_consis_loss = hyperparameters[args.env_name]['scale_consis_loss']

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    if args.lr == -1:
        args.lr = hyperparameters[args.env_name]['lr']
    if args.eta == -1:
        args.eta = hyperparameters[args.env_name]['eta']
    if args.T == -1:
        args.T = hyperparameters[args.env_name]['T']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.q_norm = hyperparameters[args.env_name]['q_norm']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    if args.gn == -1:
        args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # if args.wandb_activate:
    #     args.wandb_project = 'consistency-rl-offline'
    #     args.wandb_group = f'{args.exp}'
    #     args.wandb_name = f'{args.env_name}_{args.algo}_{args.model}_{args.ms}_{args.exp}'
        # init_wandb(args)

    writer = SummaryWriter(f"runs/{args.env_name}_{args.algo}_{args.model}_{args.ms}_{args.exp}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|{args.model}-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
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

    train_agent(env,
                args.env_name,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                writer,
                args)
