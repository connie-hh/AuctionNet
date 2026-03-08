import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Tuple
import logging

from train_ppo.ppo_bidding_env import PpoBiddingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def np2torch(x, cast_double_to_float=True):
    """Convert numpy array to torch tensor."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if cast_double_to_float and x.dtype == torch.float64:
            x = x.float()
    return x


# ---------------------------------------------------------------------------
# Policy Network (Gaussian for continuous action)
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """
    Gaussian policy: outputs (mu, log_std) for the alpha action.
    Alpha is kept positive via softplus after sampling.
    """

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128, 
                 init_alpha: float = 100.0):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (typically 1 for alpha)
            hidden_dim: Hidden layer size
            init_alpha: Initial alpha value (default 100, typically set to CPA target)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize bias to achieve target initial alpha
        # Formula: softplus(bias) * 50 ≈ init_alpha
        # For large values, softplus(x) ≈ x, so bias ≈ init_alpha / 50
        init_bias = init_alpha / 50.0
        nn.init.constant_(self.mu_head.bias, init_bias)
        
        self.log_std = nn.Parameter(torch.ones(action_dim) * 1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mu = torch.nn.functional.softplus(self.mu_head(x)) * 50 # keep alpha > 0
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def action_distribution(self, observations: torch.Tensor):
        """Returns a torch.distributions.Normal object."""
        mu, std = self.forward(observations)
        return Normal(mu, std)

    def act(self, observations: np.ndarray, return_log_prob: bool = False):
        """
        Sample action from policy.
        
        Args:
            observations: (batch_size, state_dim) numpy array
            return_log_prob: if True, also return log probability
            
        Returns:
            actions: (batch_size, action_dim) numpy array
            log_probs: (batch_size,) numpy array (if return_log_prob=True)
        """
        observations = np2torch(observations)
        dist = self.action_distribution(observations)
        actions = dist.sample()
        actions = torch.clamp(actions, min=1e-3)  # alpha must be positive
        
        if return_log_prob:
            log_probs = dist.log_prob(actions).sum(dim=-1)
            return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()
        return actions.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Baseline (Value) Network
# ---------------------------------------------------------------------------

class BaselineNetwork(nn.Module):
    """Critic network that estimates V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, lr: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)

    def calculate_advantage(self, returns: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Calculate advantages as returns - baseline."""
        observations = np2torch(observations)
        with torch.no_grad():
            baseline = self.forward(observations).cpu().numpy()
        return returns - baseline

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray):
        """Update baseline network to minimize MSE with returns."""
        returns = np2torch(returns)
        observations = np2torch(observations)
        
        baseline_pred = self.forward(observations)
        loss = nn.functional.mse_loss(baseline_pred, returns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPO:

    def __init__(
        self,
        env: PpoBiddingEnv,
        state_dim: int = 16,
        hidden_dim: int = 128,
        lr_policy: float = 3e-4,
        lr_baseline: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        batch_size: int = 2000,
        num_batches: int = 100,
        update_freq: int = 4,
        max_ep_len: int = 48,
        cpa_penalty_coef: float = 0.01,
        save_path: str = "strategy_train_env/saved_model/ppo",
        exploration_decay: float = 0.995,
        init_alpha: float = None,  # If None, uses agent's CPA
    ):
        self.env = env
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.update_freq = update_freq
        self.max_ep_len = max_ep_len
        self.cpa_penalty_coef = cpa_penalty_coef
        self.save_path = save_path
        self.exploration_decay = exploration_decay

        # Determine initial alpha from agent's CPA if not provided
        if init_alpha is None:
            init_alpha = env.agents[env.player_index].cpa
        
        # Initialize policy and baseline
        self.policy = GaussianPolicy(state_dim, action_dim=1, hidden_dim=hidden_dim, init_alpha=init_alpha)
        self.baseline_network = BaselineNetwork(state_dim, hidden_dim=hidden_dim, lr=lr_baseline)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

        logger.info(f"Initialized PPO agent:")
        logger.info(f"  Player index: {env.player_index}")
        logger.info(f"  Budget: ${env.agents[env.player_index].budget}")
        logger.info(f"  CPA target: ${env.agents[env.player_index].cpa}")
        logger.info(f"  Initial alpha: {init_alpha:.1f}")

    def sample_path(self, num_episodes: int = None):
        """Sample trajectories from the environment."""
        episode = 0
        episode_rewards = []
        episode_costs = []
        paths = []
        t = 0

        while num_episodes or t < self.batch_size:
            state = self.env.reset()
            states, actions, old_logprobs, rewards = [], [], [], []
            episode_reward = 0
            episode_cost = 0

            for step in range(self.max_ep_len):
                states.append(state)
                
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob=True)
                action, old_logprob = action[0, 0], old_logprob[0]
                
                next_state, reward, done, info = self.env.step(action)
                
                agent = self.env.agents[self.env.player_index]
                shaped_reward = self._shape_reward(reward, info["cost"], agent.cpa)
                
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(shaped_reward)
                episode_reward += reward
                episode_cost += info["cost"]
                
                state = next_state
                t += 1
                
                if done or step == self.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    episode_costs.append(episode_cost)
                    break
                if (not num_episodes) and t == self.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs)
            }
            paths.append(path)
            episode += 1
            
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards, episode_costs

    def get_returns(self, paths: List[dict]) -> np.ndarray:
        """Compute discounted returns."""
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.zeros_like(rewards)
            running_return = 0
            
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.gamma * running_return
                returns[t] = running_return
                
            all_returns.append(returns)
        
        return np.concatenate(all_returns)

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """PPO policy update with clipped objective."""
        observations = np2torch(observations)
        actions = np2torch(actions).unsqueeze(-1)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        action_dist = self.policy.action_distribution(observations)
        new_logprobs = action_dist.log_prob(actions).squeeze(-1)
        
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        """Main training loop."""
        os.makedirs(self.save_path, exist_ok=True)
        
        averaged_total_rewards = []
        averaged_total_costs = []

        logger.info("="*60)
        logger.info("Starting PPO Training")
        logger.info(f"Batches: {self.num_batches}, Batch size: {self.batch_size}")
        logger.info("="*60)

        for t in range(self.num_batches):
            logger.info(f"\nBATCH {t+1}/{self.num_batches}")
            
            paths, total_rewards, total_costs = self.sample_path()
            
            observations = np.concatenate([p["observation"] for p in paths])
            actions = np.concatenate([p["action"] for p in paths])
            old_logprobs = np.concatenate([p["old_logprobs"] for p in paths])

            returns = self.get_returns(paths)
            advantages = self.baseline_network.calculate_advantage(returns, observations)

            for _ in range(self.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, old_logprobs)

            avg_reward = np.mean(total_rewards)
            avg_cost = np.mean(total_costs)
            avg_action = np.mean(actions)
            
            averaged_total_rewards.append(avg_reward)
            averaged_total_costs.append(avg_cost)

            with torch.no_grad():
                self.policy.log_std.data *= self.exploration_decay
                current_std = self.policy.log_std.exp().item()
            
            logger.info(f"Reward: {avg_reward:.2f}, Cost: ${avg_cost:.2f}, Alpha: {avg_action:.2f}, Std: {current_std:.4f}")
            
            if (t + 1) % 10 == 0:
                self._save_checkpoint(t + 1)

        self._save_checkpoint("final")
        np.save(os.path.join(self.save_path, "rewards.npy"), averaged_total_rewards)
        np.save(os.path.join(self.save_path, "costs.npy"), averaged_total_costs)
        
        logger.info("\nTraining Complete!")
        return averaged_total_rewards, averaged_total_costs

    def _shape_reward(self, conversions: float, cost: float, cpa_target: float) -> float:
        """Reward shaping with CPA penalty."""
        if conversions > 0:
            actual_cpa = cost / conversions
            violation = max(0.0, actual_cpa - cpa_target)
            penalty = self.cpa_penalty_coef * violation
        else:
            penalty = self.cpa_penalty_coef * cost
        return conversions - penalty

    def _save_checkpoint(self, tag):
        """Save model checkpoints."""
        torch.save(self.policy.state_dict(), 
                  os.path.join(self.save_path, f"ppo_policy_{tag}.pt"))
        torch.save(self.baseline_network.state_dict(),
                  os.path.join(self.save_path, f"ppo_baseline_{tag}.pt"))


if __name__ == "__main__":
    env = PpoBiddingEnv(player_index=0)
    
    agent = PPO(
        env=env,
        state_dim=16,
        hidden_dim=128,
        num_batches=100,
        save_path="saved_model/ppo",
    )
    
    agent.train()