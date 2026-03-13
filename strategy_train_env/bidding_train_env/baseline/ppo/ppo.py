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
    """PPO agent with checkpoint loading support."""

    def __init__(
        self,
        env: PpoBiddingEnv,
        state_dim: int = 16,
        hidden_dim: int = 128,
        lr_policy: float = 3e-4,
        lr_baseline: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        batch_size: int = 2016, # 42 * max_ep_len
        num_batches: int = 100,
        update_freq: int = 8,
        max_ep_len: int = 48,
        cpa_penalty_coef: float = 0.01,
        save_path: str = "strategy_train_env/saved_model/ppo",
        exploration_decay: float = 0.995,
        init_alpha: float = None,
        load_checkpoint: str = None,  # Path to checkpoint (e.g., "saved_model/ppo/ppo_policy_50.pt")
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

        # Load checkpoint if provided
        self.start_batch = 0
        if load_checkpoint is not None:
            self.start_batch = self._load_checkpoint(load_checkpoint)

        logger.info(f"Initialized PPO agent:")
        logger.info(f"  Player index: {env.player_index}")
        logger.info(f"  Budget: ${env.agents[env.player_index].budget}")
        logger.info(f"  CPA target: ${env.agents[env.player_index].cpa}")
        logger.info(f"  Initial alpha: {init_alpha:.1f}")
        if load_checkpoint:
            logger.info(f"  Loaded checkpoint from: {load_checkpoint}")
            logger.info(f"  Resuming from batch: {self.start_batch}")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load policy and baseline from checkpoint.
        
        Args:
            checkpoint_path: Path to policy checkpoint file
            
        Returns:
            Batch number to resume from (extracted from filename)
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        # Load policy
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        logger.info(f"Loaded policy from {checkpoint_path}")
        
        # Try to load corresponding baseline
        baseline_path = checkpoint_path.replace("policy", "baseline")
        if os.path.exists(baseline_path):
            self.baseline_network.load_state_dict(torch.load(baseline_path, map_location='cpu'))
            logger.info(f"Loaded baseline from {baseline_path}")
        else:
            logger.warning(f"Baseline checkpoint not found: {baseline_path}")
        
        # Extract batch number from filename (e.g., "ppo_policy_50.pt" -> 50)
        import re
        match = re.search(r'_(\d+)\.pt$', checkpoint_path)
        if match:
            batch_num = int(match.group(1))
            for name, attr in [("rewards", "averaged_total_rewards"), 
                   ("costs", "averaged_total_costs"), 
                   ("alphas", "averaged_total_actions")]:
                npy_path = os.path.join(self.save_path, f"{name}.npy")
                if os.path.exists(npy_path):
                    setattr(self, attr, list(np.load(npy_path)))
                    logger.info(f"Loaded {name} history ({len(getattr(self, attr))} entries)")
                else:
                    setattr(self, attr, [])
            return batch_num
        elif 'final' in checkpoint_path:
            logger.warning("Loaded 'final' checkpoint - cannot resume training")
            return 0
        return 0

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

            cumulative_value = 0
            cumulative_cost = 0

            for step in range(self.max_ep_len):
                states.append(state)
                
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob=True)
                action, old_logprob = action[0, 0], old_logprob[0]

                # Store state before this timestep
                cumulative_value_before = cumulative_value
                cumulative_cost_before = cumulative_cost

                # Step environment and update cumulative
                next_state, reward, done, info = self.env.step(action)

                if reward > 0:
                    cumulative_value += info["value"]
                    cumulative_cost += info["cost"]

                # Calculate CPAs
                cpa_before = cumulative_cost_before / cumulative_value_before if cumulative_value_before > 0 else 0
                cpa_after = cumulative_cost / cumulative_value if cumulative_value > 0 else 0

                agent = self.env.agents[self.env.player_index]
                # print(f'Current action cost: {info["cost"]}; value: {info["value"]}; cpa: {info["cost"] / info["value"]}')

                # Penalty: only if over target AND didn't improve
                if cpa_after > agent.cpa and cpa_after > cpa_before:
                    violation = cpa_after - agent.cpa
                    penalty = self.cpa_penalty_coef * violation
                    # print(f'Current action cost: {info["cost"]}; value: {info["value"]}; cpa: {info["cost"] / info["value"]}')
                    # print(f'CPA before: {cpa_before}; CPA after: {cpa_after}; CPA penalty: {penalty}')
                else:
                    penalty = 0

                shaped_reward = reward - penalty
                # print(f'Shaped reward: {reward - penalty}')
                
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
        
        if not hasattr(self, 'averaged_total_rewards'):
            averaged_total_rewards = []
            averaged_total_costs = []
            averaged_total_actions = []
        else:
            averaged_total_rewards = self.averaged_total_rewards
            averaged_total_costs = self.averaged_total_costs
            averaged_total_actions = self.averaged_total_actions
        print(averaged_total_rewards)

        logger.info("="*60)
        logger.info("Starting PPO Training")
        logger.info(f"Batches: {self.num_batches}, Batch size: {self.batch_size}")
        if self.start_batch > 0:
            logger.info(f"Resuming from batch: {self.start_batch}")
        logger.info("="*60)

        for t in range(self.start_batch, self.num_batches):
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
            averaged_total_actions.append(avg_action)

            with torch.no_grad():
                self.policy.log_std.data *= self.exploration_decay
                current_std = self.policy.log_std.exp().item()
            
            logger.info(f"Reward: {avg_reward:.2f}, Cost: ${avg_cost:.2f}, Alpha: {avg_action:.2f}, Std: {current_std:.4f}")
            
            if (t + 1) % 10 == 0:
                self._save_checkpoint(t + 1)

        self._save_checkpoint("final")
        np.save(os.path.join(self.save_path, "rewards.npy"), averaged_total_rewards)
        np.save(os.path.join(self.save_path, "costs.npy"), averaged_total_costs)
        np.save(os.path.join(self.save_path, "alphas.npy"), averaged_total_actions)
        
        logger.info("\nTraining Complete!")
        return averaged_total_rewards, averaged_total_costs

    # def _shape_reward(self, conversions: float, cost: float, cpa_target: float) -> float:
    #     """Reward shaping with CPA penalty."""
    #     if conversions > 0:
    #         actual_cpa = cost / conversions
    #         violation = max(0.0, actual_cpa - cpa_target)
    #         penalty = self.cpa_penalty_coef * violation
    #     else:
    #         penalty = self.cpa_penalty_coef * cost
    #     return conversions - penalty

    def _save_checkpoint(self, tag):
        """Save model checkpoints."""
        torch.save(self.policy.state_dict(), 
                  os.path.join(self.save_path, f"ppo_policy_{tag}.pt"))
        torch.save(self.baseline_network.state_dict(),
                  os.path.join(self.save_path, f"ppo_baseline_{tag}.pt"))


if __name__ == "__main__":
    env = PpoBiddingEnv(player_index=0)
    
    # Resume from checkpoint
    agent = PPO(
        env=env,
        hidden_dim=128,
        load_checkpoint="saved_model_no_cpa_v2/ppo/ppo_policy_300.pt",
        num_batches=500,
        save_path="saved_model_no_cpa_v2/ppo"
    )
    
    # # Train from scratch
    # agent = PPO(
    #     env=env,
    #     state_dim=16,
    #     hidden_dim=128,
    #     num_batches=300,
    #     save_path="saved_model_no_cpa_v2/ppo",
    # )
    
    agent.train()