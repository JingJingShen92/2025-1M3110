import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import collections
import random
from tqdm import tqdm
import imageio
import argparse
from pathlib import Path
#
# # Ignore harmless warnings from Gymnasium
# warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

def save_gif(frames, path, fps=30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, duration=1 / fps)
# --- 1. Replay Buffer ---
# Used to store agent-environment interaction experiences (s, a, r, s', done)
class ReplayBuffer:
    def __init__(self, capacity):
        # Use collections.deque to implement a fixed-size queue
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Store a single interaction experience into the replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from the replay buffer"""
        transitions = random.sample(self.buffer, batch_size)
        # Unpack the experience tuples and convert to NumPy arrays
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        """Return the current number of experiences in the replay buffer"""
        return len(self.buffer)


# --- 2. Network Definitions ---

class Actor(nn.Module):
    """
    Policy Network / Actor
    Input: state
    Output: mean and std of Gaussian distribution for action, and log probability of the action
    """

    def __init__(self, state_dim, action_dim, hidden_dim, action_high, device):
        super(Actor, self).__init__()
        self.device = device
        # Convert the action upper bound to tensor
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        # Used to limit the range of log_std to prevent numerical instability
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clip log_std to specified range to ensure training stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Create Gaussian distribution
        dist = Normal(mean, std)

        if deterministic:
            # In evaluation mode, use the mean as the deterministic action
            z = mean
        else:
            # In training mode, use reparameterization trick to sample from the distribution
            z = dist.rsample()

        # Use tanh to compress the action into [-1, 1]
        action = torch.tanh(z)

        # Compute log probability, corrected for tanh transformation
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdim=True)

        # Scale action from [-1, 1] to the actual action range of the environment
        scaled_action = action * self.action_high

        return scaled_action, log_prob


class Critic(nn.Module):
    """
    Value Network / Critic
    Input: state and action
    Output: Q value for the state-action pair (Twin Critic structure)
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Q2 network (Twin Critic)
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate state and action as input
        sa = torch.cat([state, action], 1)

        # Compute Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Compute Q2
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


# --- 3. SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim, action_high, hidden_dim, lr_actor, lr_critic, lr_alpha, gamma, tau,
                 device):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Initialize Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_high, device).to(device)

        # Initialize Critic networks and their target networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic entropy regularization coefficient alpha
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32, device=device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp().item()

    def select_action(self, state, deterministic=False):
        """Select action based on current state"""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(state, deterministic)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        """Sample data from the replay buffer and update networks"""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            target_q = rewards + (1 - dones) * self.gamma * (min_q_target - self.alpha * next_log_prob)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        new_actions, log_prob = self.actor(states)
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path, step):
        """Save model weights by step, will not overwrite"""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.critic.state_dict(), f"{path}/critic_{step}.pth")
        torch.save(self.actor.state_dict(), f"{path}/actor_{step}.pth")

    def load(self, path, step):
        """Load model weights"""
        if step is None:
            self.critic.load_state_dict(torch.load(f"{path}/critic.pth", map_location=self.device))
            self.actor.load_state_dict(torch.load(f"{path}/actor.pth", map_location=self.device))
        else:
            self.critic.load_state_dict(torch.load(f"{path}/critic_{step}.pth", map_location=self.device))
            self.actor.load_state_dict(torch.load(f"{path}/actor_{step}.pth", map_location=self.device))
        # Sync target network after loading
        self.critic_target.load_state_dict(self.critic.state_dict())

# --- 4. Main Function ---
def main(args):
    print(f"Device: {args.device}")

    # --- Training mode ---
    if not args.eval:
        print("Mode: Training")
        env = gym.make(args.env_name)

        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high

        agent = SACAgent(state_dim, action_dim, action_high, args.hidden_dim,
                         args.lr_actor, args.lr_critic, args.lr_alpha,
                         args.gamma, args.tau, args.device)

        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        state, _ = env.reset(seed=args.seed)
        episode_reward = 0
        episode_num = 0

        pbar = tqdm(range(args.max_steps))
        for step in pbar:
            if step < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                episode_num += 1
                pbar.set_postfix({
                    "Episode": episode_num,
                    "Reward": f"{episode_reward:.2f}",
                    "Alpha": f"{agent.alpha:.4f}"
                })
                state, _ = env.reset()
                episode_reward = 0

            if len(replay_buffer) > args.batch_size and step >= args.start_steps:
                agent.update(replay_buffer, args.batch_size)

            if (step + 1) % args.save_interval == 0:
                agent.save(args.model_dir, step + 1)
                tqdm.write(f"\nModel saved at: {args.model_dir}/[actor|critic]_{step + 1}.pth")

        env.close()
        agent.save(args.model_dir, args.max_steps)
        print(f"Training finished. Final model saved at: {args.model_dir}/[actor|critic]_{args.max_steps}.pth")


    # --- Evaluation mode ---
    else:
        print("Mode: Evaluation")
        # In evaluation mode, add render_mode='human' for visualization
        env = gym.make(args.env_name, render_mode='rgb_array', max_episode_steps=args.max_episode_steps)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high

        agent = SACAgent(state_dim, action_dim, action_high, args.hidden_dim,
                         args.lr_actor, args.lr_critic, args.lr_alpha,
                         args.gamma, args.tau, args.device)

        try:
            agent.load(args.model_dir, args.load_model_step)
            print(f"Model loaded from {args.model_dir}.")
        except FileNotFoundError:
            print(f"Error: No model files found in {args.model_dir}. Please train first.")
            return

        for i in range(args.eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            frames = []
            while not done:
                # Render and append each frame as an RGB array
                frame = env.render()  # Returns RGB array by default in Gymnasium
                frames.append(frame)
                # Use deterministic action during evaluation
                action = agent.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated or len(frames) >= 300
                state = next_state
                episode_reward += reward

            print(f"Evaluation Episode {i + 1}/{args.eval_episodes} | Reward: {episode_reward:.2f}")

            # Save GIF for each episode
            save_gif(frames, f"figures/eval_{args.load_model_step}.gif", fps=30)

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic")

    # Mode and path arguments
    parser.add_argument("--eval", action="store_true", help="Set evaluation mode")
    parser.add_argument("--model-dir", default="./models", type=str, help="Directory to save/load models")
    parser.add_argument("--load-model-step", default=None, type=int, help="Specific model step to load (default: latest)")

    # Environment and seed arguments
    parser.add_argument("--env-name", default="HalfCheetah-v5", type=str, help="Gymnasium environment name")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--max-episode-steps", default=1000, type=int, help="Maximum episode steps during inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Compute device (cuda/cpu)")

    # Training hyperparameters
    parser.add_argument("--max-steps", default=1_000_000, type=int, help="Maximum training steps")
    parser.add_argument("--start-steps", default=10_000, type=int, help="Initial random exploration steps")
    parser.add_argument("--replay-buffer-size", default=1_000_000, type=int, help="Replay buffer size")
    parser.add_argument("--batch-size", default=256, type=int, help="Batch size")
    parser.add_argument("--hidden-dim", default=256, type=int, help="Hidden dimension of neural network")

    # SAC algorithm parameters
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--tau", default=0.005, type=float, help="Soft update coefficient for target network")
    parser.add_argument("--lr-actor", default=3e-4, type=float, help="Actor learning rate")
    parser.add_argument("--lr-critic", default=3e-4, type=float, help="Critic learning rate")
    parser.add_argument("--lr-alpha", default=3e-4, type=float, help="Alpha learning rate")

    # Saving and evaluation parameters
    parser.add_argument("--save-interval", default=50_000, type=int, help="Model saving interval steps")
    parser.add_argument("--eval-episodes", default=1, type=int, help="Number of episodes in evaluation mode")

    args = parser.parse_args()

    main(args)
