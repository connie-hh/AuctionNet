import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Tuple
import logging

from train_ppo.two_agent_ppo_bidding_env import TwoAgentPpoBiddingEnv
from strategy_train_env.bidding_train_env.baseline.ppo.ppo import np2torch, GaussianPolicy, BaselineNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Two-Agent PPO Trainer
# ---------------------------------------------------------------------------

class TwoAgentPPO:
    """
    Two-agent PPO trainer with independent policies.
    Each agent has its own policy, baseline, and optimizer.
    """

    def __init__(
        self,
        env: TwoAgentPpoBiddingEnv,
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
        save_path: str = "strategy_train_env/saved_model/two_agent_ppo",
        exploration_decay: float = 0.995,
    ):
        self.env = env
        self.num_agents = 2
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

        # Create separate policy and baseline for each agent
        self.policies = [
            GaussianPolicy(state_dim, action_dim=1, hidden_dim=hidden_dim)
            for _ in range(self.num_agents)
        ]
        
        self.baseline_networks = [
            BaselineNetwork(state_dim, hidden_dim=hidden_dim, lr=lr_baseline)
            for _ in range(self.num_agents)
        ]
        
        self.optimizers = [
            optim.Adam(policy.parameters(), lr=lr_policy)
            for policy in self.policies
        ]

    def sample_path(self):
        """
        Collect rollouts for both agents.
        
        Returns:
            paths_per_agent: [[agent0_paths], [agent1_paths]]
            episode_rewards_per_agent: [[agent0_rewards], [agent1_rewards]]
            episode_costs_per_agent: [[agent0_costs], [agent1_costs]]
        """
        paths_per_agent = [[], []]
        episode_rewards_per_agent = [[], []]
        episode_costs_per_agent = [[], []]
        
        total_timesteps = 0
        episode = 0
        
        while total_timesteps < self.batch_size:
            # Reset environment - returns [state0, state1]
            states = self.env.reset()
            
            # Initialize episode storage for each agent
            for agent_idx in range(self.num_agents):
                paths_per_agent[agent_idx].append({
                    "observation": [],
                    "action": [],
                    "reward": [],
                    "old_logprobs": []
                })
            
            episode_rewards = [0.0, 0.0]
            episode_costs = [0.0, 0.0]
            
            for step in range(self.max_ep_len):
                # Get actions from each agent's policy
                actions = []
                log_probs = []
                
                for agent_idx in range(self.num_agents):
                    action, log_prob = self.policies[agent_idx].act(
                        states[agent_idx][None], 
                        return_log_prob=True
                    )
                    actions.append(action[0, 0])  # Scalar
                    log_probs.append(log_prob[0])
                
                # Step environment
                next_states, rewards, dones, infos = self.env.step(actions)
                
                # Shape rewards for each agent
                shaped_rewards = []
                for agent_idx in range(self.num_agents):
                    agent = self.env.agents[agent_idx]
                    shaped_reward = self._shape_reward(
                        rewards[agent_idx], 
                        infos[agent_idx]['cost'], 
                        agent.cpa
                    )
                    shaped_rewards.append(shaped_reward)
                
                # Store transitions for each agent
                for agent_idx in range(self.num_agents):
                    paths_per_agent[agent_idx][-1]["observation"].append(states[agent_idx])
                    paths_per_agent[agent_idx][-1]["action"].append(actions[agent_idx])
                    paths_per_agent[agent_idx][-1]["reward"].append(shaped_rewards[agent_idx])
                    paths_per_agent[agent_idx][-1]["old_logprobs"].append(log_probs[agent_idx])
                    
                    episode_rewards[agent_idx] += rewards[agent_idx]
                    episode_costs[agent_idx] += infos[agent_idx]['cost']
                
                states = next_states
                total_timesteps += 1
                
                # Check if any agent is done
                if any(dones):
                    for agent_idx in range(self.num_agents):
                        episode_rewards_per_agent[agent_idx].append(episode_rewards[agent_idx])
                        episode_costs_per_agent[agent_idx].append(episode_costs[agent_idx])
                    break
                
                if total_timesteps >= self.batch_size:
                    break
            
            episode += 1
        
        # Convert lists to arrays for each agent
        for agent_idx in range(self.num_agents):
            for path in paths_per_agent[agent_idx]:
                path["observation"] = np.array(path["observation"])
                path["action"] = np.array(path["action"])
                path["reward"] = np.array(path["reward"])
                path["old_logprobs"] = np.array(path["old_logprobs"])
        
        return paths_per_agent, episode_rewards_per_agent, episode_costs_per_agent

    def get_returns(self, paths: List[dict]) -> np.ndarray:
        """Compute discounted returns for each timestep."""
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

    def update_policy(self, agent_idx: int, observations: np.ndarray, actions: np.ndarray, 
                     advantages: np.ndarray, old_logprobs: np.ndarray):
        """Update a specific agent's policy using PPO clipped objective."""
        observations = np2torch(observations)
        actions = np2torch(actions).unsqueeze(-1)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        # Compute new log probs
        action_dist = self.policies[agent_idx].action_distribution(observations)
        new_logprobs = action_dist.log_prob(actions).squeeze(-1)
        
        # PPO clipped objective
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean()
        
        # Update
        self.optimizers[agent_idx].zero_grad()
        loss.backward()
        self.optimizers[agent_idx].step()
        
        return loss.item()

    def train(self):
        """Main training loop."""
        os.makedirs(self.save_path, exist_ok=True)
        
        # Track metrics per agent
        all_rewards_per_agent = [[], []]
        all_costs_per_agent = [[], []]
        all_alphas_per_agent = [[], []]  # Track alpha values
        
        logger.info("="*60)
        logger.info("Starting Two-Agent PPO Training")
        logger.info(f"Number of agents: {self.num_agents}")
        logger.info(f"Total batches: {self.num_batches}")
        logger.info(f"Batch size: {self.batch_size} timesteps")
        logger.info("="*60)
        
        for batch_idx in range(self.num_batches):
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_idx+1}/{self.num_batches}")
            logger.info(f"{'='*60}")
            
            # Collect rollouts for both agents
            paths_per_agent, episode_rewards_per_agent, episode_costs_per_agent = self.sample_path()
            
            num_episodes = len(episode_rewards_per_agent[0])
            logger.info(f"Collected {num_episodes} episodes")
            
            # Update each agent independently
            for agent_idx in range(self.num_agents):
                logger.info(f"\n--- Updating Agent {agent_idx} ---")
                
                paths = paths_per_agent[agent_idx]
                observations = np.concatenate([p["observation"] for p in paths])
                actions = np.concatenate([p["action"] for p in paths])
                rewards = np.concatenate([p["reward"] for p in paths])
                old_logprobs = np.concatenate([p["old_logprobs"] for p in paths])
                
                # Compute returns and advantages
                returns = self.get_returns(paths)
                advantages = self.baseline_networks[agent_idx].calculate_advantage(returns, observations)
                
                # Update policy and baseline
                policy_losses = []
                baseline_losses = []
                for _ in range(self.update_freq):
                    baseline_loss = self.baseline_networks[agent_idx].update_baseline(returns, observations)
                    policy_loss = self.update_policy(agent_idx, observations, actions, advantages, old_logprobs)
                    policy_losses.append(policy_loss)
                    baseline_losses.append(baseline_loss)
                
                # Decay exploration
                with torch.no_grad():
                    self.policies[agent_idx].log_std.data *= self.exploration_decay
                    current_std = self.policies[agent_idx].log_std.exp().item()
                
                # Log agent stats
                avg_reward = np.mean(episode_rewards_per_agent[agent_idx])
                std_reward = np.std(episode_rewards_per_agent[agent_idx])
                avg_cost = np.mean(episode_costs_per_agent[agent_idx])
                avg_action = np.mean(actions)
                
                all_rewards_per_agent[agent_idx].append(avg_reward)
                all_costs_per_agent[agent_idx].append(avg_cost)
                all_alphas_per_agent[agent_idx].append(avg_action)  # Save average alpha
                
                logger.info(f"Reward:          {avg_reward:7.2f} ± {std_reward:6.2f}")
                logger.info(f"Cost:            {avg_cost:7.2f}")
                logger.info(f"Policy Loss:     {np.mean(policy_losses):7.4f}")
                logger.info(f"Baseline Loss:   {np.mean(baseline_losses):7.4f}")
                logger.info(f"Action (alpha):  {avg_action:7.4f}")
                logger.info(f"Exploration std: {current_std:7.4f}")
            
            # Save checkpoints
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"\nSaving checkpoint at batch {batch_idx + 1}...")
                self._save_checkpoint(batch_idx + 1)
        
        # Final save
        self._save_checkpoint("final")
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        
        # Save reward histories
        for agent_idx in range(self.num_agents):
            np.save(
                os.path.join(self.save_path, f"agent{agent_idx}_rewards.npy"),
                all_rewards_per_agent[agent_idx]
            )
            np.save(
                os.path.join(self.save_path, f"agent{agent_idx}_costs.npy"),
                all_costs_per_agent[agent_idx]
            )
            np.save(
                os.path.join(self.save_path, f"agent{agent_idx}_alphas.npy"),
                all_alphas_per_agent[agent_idx]
            )
        
        # Final statistics
        logger.info(f"\nFinal Statistics:")
        for agent_idx in range(self.num_agents):
            logger.info(f"\nAgent {agent_idx}:")
            logger.info(f"  Average Reward (last 10): {np.mean(all_rewards_per_agent[agent_idx][-10:]):7.2f}")
            logger.info(f"  Average Cost (last 10):   {np.mean(all_costs_per_agent[agent_idx][-10:]):7.2f}")
            logger.info(f"  Average Alpha (last 10):  {np.mean(all_alphas_per_agent[agent_idx][-10:]):7.2f}")
            logger.info(f"  Best Reward: {np.max(all_rewards_per_agent[agent_idx]):7.2f}")
        
        return all_rewards_per_agent, all_costs_per_agent, all_alphas_per_agent

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
        """Save all agents' policies and baselines."""
        for agent_idx in range(self.num_agents):
            torch.save(
                self.policies[agent_idx].state_dict(),
                os.path.join(self.save_path, f"agent{agent_idx}_policy_{tag}.pt")
            )
            torch.save(
                self.baseline_networks[agent_idx].state_dict(),
                os.path.join(self.save_path, f"agent{agent_idx}_baseline_{tag}.pt")
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = TwoAgentPpoBiddingEnv()
    
    trainer = TwoAgentPPO(
        env=env,
        state_dim=16,
        hidden_dim=128,
        lr_policy=3e-4,
        lr_baseline=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        batch_size=2000,
        num_batches=1000,
        update_freq=4,
        max_ep_len=48,
        cpa_penalty_coef=0.0,
        save_path="strategy_train_env/saved_model/two_agent_ppo",
    )
    
    rewards_per_agent, costs_per_agent, alphas_per_agent = trainer.train()
    
    print(f"\nFinal Performance:")
    print(f"Agent 0: Avg Reward={np.mean(rewards_per_agent[0][-10:]):.2f}, Avg Alpha={np.mean(alphas_per_agent[0][-10:]):.2f}")
    print(f"Agent 1: Avg Reward={np.mean(rewards_per_agent[1][-10:]):.2f}, Avg Alpha={np.mean(alphas_per_agent[1][-10:]):.2f}")