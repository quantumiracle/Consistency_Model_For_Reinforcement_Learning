import copy, math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger


from agents.consistency import Consistency
from agents.model import MLP, Unet, LN_Resnet
from agents.helpers import EMA, kerras_boundaries
from agents.ql_diffusion import Critic

class ExponentialScheduler():
    def __init__(self, v_start=1.5, v_final=0.2, decay_steps=2*10**2):
        """A scheduler for exponential decay.

        :param v_start: starting value of epsilon, default 1. as purely random policy 
        :type v_start: float
        :param v_final: final value of epsilon
        :type v_final: float
        :param decay_steps: number of steps from eps_start to eps_final
        :type decay_steps: int
        """
        self.v_start = v_start
        self.v_final = v_final
        self.decay_steps = decay_steps
        self.value = self.v_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self):
        """
        The choice of eps_decay:
        ------------------------
        start = 1
        final = 0.01
        decay = 10**6  # the decay steps can be 1/10 over all steps 10000*1000
        final + (start-final)*np.exp(-1*(10**7)/decay)
        
        => 0.01

        """
        self.current_frame_idx += 1
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.value = self.v_final + (self.v_start - self.v_final) * math.exp(-1. * delta_frame_idx / self.decay_steps)
        return self.value

    def get_value(self, ):
        return self.value

# https://github.com/Kinyugo/consistency_models/blob/main/consistency_models/consistency_models.py
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1


def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1

    return int(num_timesteps)


def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps

# https://github.com/Kinyugo/consistency_models/blob/main/consistency_models/consistency_models.py
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    """Implements the proposed timestep discretization schedule.
    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.
    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1


def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
) -> int:
    """Implements the improved timestep discretization schedule.
    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.
    Returns
    -------
    int
        Number of timesteps at the current point in training.
    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1

    return int(num_timesteps)


def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.
    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.
    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.
    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps

class Consistency_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 q_norm=False,
                 adaptive_ema=False,
                 scale_consis_loss=True,
                 steps_per_epoch=1000,
                 improved_CT=False,  # [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
                 ):

        self.device = device
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        # self.model = LN_Resnet(state_dim=state_dim, action_dim=action_dim, device=device)

        if improved_CT:
            loss_type = 'pseudo_huber'
        else:
            loss_type = 'l2'
        self.actor = Consistency(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               n_timesteps=n_timesteps, loss_type=loss_type).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.q_norm = q_norm
        self.scale_consis_loss = scale_consis_loss
        self.improved_CT = improved_CT

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_actor = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        self.adaptive_ema = adaptive_ema
        self.steps_per_epoch = steps_per_epoch

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_actor, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        loss_ema = None

        for itr in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_actor(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_actor(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()
            
            # normalize the target_q
            if self.q_norm:
                target_q = (target_q) / (target_q.std() + 1e-6)

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2) # return grad value is before clipped, although the grad for update is clipped
            self.critic_optimizer.step()

            # loss for Q-learning
            new_action = self.actor(state)
            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            """ Policy Training """

            if len(state) > 0:
                # loss for BC with consistency model
                if self.improved_CT:
                    N = improved_timesteps_schedule(itr, iterations, initial_timesteps=10, final_timesteps=1280)
                else:
                    # N = math.ceil(math.sqrt((itr * (150**2 - 4) / iterations) + 4) - 1) + 1  # s0=2, s1=150
                    N = timesteps_schedule(itr, iterations, initial_timesteps=2, final_timesteps=150) # eqivalent to above

                if self.adaptive_ema:
                    start_scales = 2.0  # s0
                    c = np.log(self.ema_decay) * start_scales  # https://github.com/openai/consistency_models/blob/6d26080c58244555c031dbc63080c0961af74200/cm/script_util.py#L195
                    target_ema = np.exp(c / N)
                boundaries = kerras_boundaries(7.0, 0.002, N, self.actor.max_T).to(self.device)

                z = torch.randn_like(action)
                if self.improved_CT:
                    t = lognormal_timestep_distribution(action.shape[0], boundaries, mean=-1.1, std=2.0)
                    t = t.view(-1, 1).to(self.device)
                else:
                    t = torch.randint(0, N - 1, (action.shape[0], 1), device=self.device)
                t_1 = boundaries[t]
                t_2 = boundaries[t + 1]

                if self.improved_CT:
                    teacher_model = None  # same as self.actor
                else:
                    teacher_model = self.ema_actor

                bc_loss = self.actor.loss(state, action, z, t_1, t_2, teacher_model)
            else:
                bc_loss = torch.zeros((1,), device=self.device)
            
            mean_bc_loss = bc_loss.mean()
            if loss_ema is None:
                loss_ema = mean_bc_loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * mean_bc_loss.item()

            if self.scale_consis_loss:
                delta_t = t_2 - t_1 # shape: (batch_size, 1)
                mean_bc_loss = (100./delta_t * bc_loss).mean()  # this is a simple fix for scale issue; there maybe finer version
            actor_loss = mean_bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                if self.adaptive_ema:
                    self.ema.set(target_ema)
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('BC Loss', mean_bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(mean_bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None, map_location=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location=map_location))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth', map_location=map_location))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location=map_location))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth', map_location=map_location))

            

