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

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable, state-independent

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
        """
        Calculate advantages as returns - baseline.
        
        Args:
            returns: (batch_size,) array of discounted returns
            observations: (batch_size, state_dim) array
            
        Returns:
            advantages: (batch_size,) array
        """
        observations = np2torch(observations)
        with torch.no_grad():
            baseline = self.forward(observations).cpu().numpy()
        return returns - baseline

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray):
        """
        Update baseline network to minimize MSE with returns.
        
        Args:
            returns: (batch_size,) array
            observations: (batch_size, state_dim) array
            
        Returns:
            loss: scalar baseline loss value
        """
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
    """
    PPO agent adapted from the reference implementation.
    Follows the architecture from the provided PPO code.
    """

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
        cpa_penalty_coef: float = 0.1,
        save_path: str = "saved_model/ppo",
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

        # Initialize policy and baseline
        self.policy = GaussianPolicy(state_dim, action_dim=1, hidden_dim=hidden_dim)
        self.baseline_network = BaselineNetwork(state_dim, hidden_dim=hidden_dim, lr=lr_baseline)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

        self.episode_rewards = []
        self.batch_rewards = []

    def sample_path(self, num_episodes: int = None):
        """
        Sample paths (trajectories) from the environment.
        
        Args:
            num_episodes: number of episodes to sample. If None, sample until batch_size reached.
            
        Returns:
            paths: list of dicts with keys ['observation', 'action', 'reward', 'old_logprobs']
            total_rewards: list of total rewards per episode
        """
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
                
                # Get action and log prob from policy
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob=True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0, 0], old_logprob[0]  # scalar action
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Shape reward with CPA penalty
                agent = self.env.agents[self.env.player_index]
                shaped_reward = self._shape_reward(reward, info["cost"], agent.cpa)
                
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(shaped_reward)
                episode_reward += reward  # track raw reward for logging
                episode_cost += info["cost"]
                
                state = next_state
                t += 1
                
                if done or step == self.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    episode_costs.append(episode_cost)
                    logger.debug(f"  Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Steps={step+1}")
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
        """
        Compute discounted returns for each timestep.
        
        Args:
            paths: list of path dicts
            
        Returns:
            all_returns: concatenated array of returns for all paths
        """
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

    def calculate_advantage(self, returns: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Calculate advantages using baseline network."""
        return self.baseline_network.calculate_advantage(returns, observations)

    def update_policy(self, observations: np.ndarray, actions: np.ndarray, 
                     advantages: np.ndarray, old_logprobs: np.ndarray):
        """
        Perform one PPO update using clipped objective.
        
        Args:
            observations: (batch_size, state_dim)
            actions: (batch_size,)
            advantages: (batch_size,)
            old_logprobs: (batch_size,)
            
        Returns:
            loss: scalar policy loss value
        """
        observations = np2torch(observations)
        actions = np2torch(actions).unsqueeze(-1)  # (batch_size, 1)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        # Compute new log probs
        action_dist = self.policy.action_distribution(observations)
        new_logprobs = action_dist.log_prob(actions).squeeze(-1)
        
        # PPO clipped objective
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        """Main training loop following reference implementation structure."""
        os.makedirs(self.save_path, exist_ok=True)
        
        all_total_rewards = []
        all_total_costs = []
        averaged_total_rewards = []
        averaged_total_costs = []

        logger.info("="*60)
        logger.info("Starting PPO Training")
        logger.info(f"Total batches: {self.num_batches}")
        logger.info(f"Batch size: {self.batch_size} timesteps")
        logger.info(f"Update frequency: {self.update_freq} passes per batch")
        logger.info(f"Max episode length: {self.max_ep_len}")
        logger.info(f"Save path: {self.save_path}")
        logger.info("="*60)

        for t in range(self.num_batches):
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {t+1}/{self.num_batches}")
            logger.info(f"{'='*60}")
            
            # Collect minibatch of samples
            logger.info("Collecting rollouts...")
            paths, total_rewards, total_costs = self.sample_path()
            all_total_rewards.extend(total_rewards)
            all_total_costs.extend(total_costs)
            
            num_episodes = len(total_rewards)
            total_timesteps = sum(len(path["observation"]) for path in paths)
            
            logger.info(f"Collected {num_episodes} episodes ({total_timesteps} timesteps)")
            
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # Compute returns and advantages
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # Run training operations
            logger.info(f"Running {self.update_freq} policy update passes...")
            policy_losses = []
            baseline_losses = []
            
            for k in range(self.update_freq):
                baseline_loss = self.baseline_network.update_baseline(returns, observations)
                policy_loss = self.update_policy(observations, actions, advantages, old_logprobs)
                policy_losses.append(policy_loss)
                baseline_losses.append(baseline_loss)
                
                if k == 0 or (k + 1) % max(1, self.update_freq // 4) == 0:
                    logger.debug(f"  Update {k+1}/{self.update_freq}: "
                               f"Policy Loss={policy_loss:.4f}, "
                               f"Baseline Loss={baseline_loss:.4f}")

            # Compute statistics
            avg_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            sigma_reward = std_reward / np.sqrt(len(total_rewards))
            
            avg_cost = np.mean(total_costs)
            std_cost = np.std(total_costs)
            
            avg_policy_loss = np.mean(policy_losses)
            avg_baseline_loss = np.mean(baseline_losses)
            
            avg_action = np.mean(actions)
            std_action = np.std(actions)
            
            averaged_total_rewards.append(avg_reward)
            averaged_total_costs.append(avg_cost)
            
            # Logging
            logger.info(f"\n{'─'*60}")
            logger.info(f"BATCH {t+1} SUMMARY:")
            logger.info(f"{'─'*60}")
            logger.info(f"Episodes:        {num_episodes}")
            logger.info(f"Total Timesteps: {total_timesteps}")
            logger.info(f"Reward:          {avg_reward:7.2f} ± {sigma_reward:6.2f} (std: {std_reward:6.2f})")
            logger.info(f"Cost:            {avg_cost:7.2f} ± {std_cost/np.sqrt(len(total_costs)):6.2f} (std: {std_cost:6.2f})")
            logger.info(f"Policy Loss:     {avg_policy_loss:7.4f}")
            logger.info(f"Baseline Loss:   {avg_baseline_loss:7.4f}")
            logger.info(f"Action (alpha):  {avg_action:7.4f} ± {std_action:6.4f}")
            logger.info(f"Returns:         mean={np.mean(returns):7.2f}, std={np.std(returns):6.2f}")
            logger.info(f"Advantages:      mean={np.mean(advantages):7.4f}, std={np.std(advantages):6.4f}")
            
            # Save checkpoint every 10 batches
            if (t + 1) % 10 == 0:
                logger.info(f"\nSaving checkpoint at batch {t+1}...")
                self._save_checkpoint(t + 1)

        # Final save
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        self._save_checkpoint("final")
        logger.info(f"Final model saved to {self.save_path}/")
        
        # Save reward history
        np.save(os.path.join(self.save_path, "rewards.npy"), averaged_total_rewards)
        np.save(os.path.join(self.save_path, "costs.npy"), averaged_total_costs)
        
        # Final statistics
        logger.info(f"\nFinal Statistics:")
        logger.info(f"  Average Reward (last 10 batches): {np.mean(averaged_total_rewards[-10:]):7.2f}")
        logger.info(f"  Average Cost (last 10 batches):   {np.mean(averaged_total_costs[-10:]):7.2f}")
        logger.info(f"  Best Reward:  {np.max(averaged_total_rewards):7.2f} (batch {np.argmax(averaged_total_rewards)+1})")
        logger.info(f"  Total Episodes Trained: {len(all_total_rewards)}")
        
        return averaged_total_rewards, averaged_total_costs

    def _shape_reward(self, conversions: float, cost: float, cpa_target: float) -> float:
        """Add CPA-violation penalty to raw conversion reward."""
        if conversions > 0:
            actual_cpa = cost / conversions
            violation = max(0.0, actual_cpa - cpa_target)
            penalty = self.cpa_penalty_coef * violation
        else:
            penalty = self.cpa_penalty_coef * cost
        return conversions - penalty

    def _save_checkpoint(self, tag):
        """Save policy and baseline network weights."""
        torch.save(
            self.policy.state_dict(),
            os.path.join(self.save_path, f"ppo_policy_{tag}.pt"),
        )
        torch.save(
            self.baseline_network.state_dict(),
            os.path.join(self.save_path, f"ppo_baseline_{tag}.pt"),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    
    env = PpoBiddingEnv(player_index=0)
    
    agent = PPO(
        env=env,
        state_dim=16,
        hidden_dim=128,
        lr_policy=3e-4,
        lr_baseline=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        batch_size=2000,
        num_batches=100,
        update_freq=4,
        max_ep_len=48,
        save_path="saved_model/ppo",
    )
    
    agent.train()