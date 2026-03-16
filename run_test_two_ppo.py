"""
Two-Agent PPO Testing Environment
Modeled after run_test.py with full tracking and analysis.
"""

import sys
import time
import os
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch

from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker
from simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis
from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from strategy_train_env.bidding_train_env.baseline.ppo.ppo import GaussianPolicy
from run.run_test import get_winner, adjust_over_cost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOAgent:
    """Wrapper for PPO policy to match the agent interface."""
    
    def __init__(self, policy_path: str, budget: float, cpa: float, category: int, 
                 name: str, state_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize PPO agent wrapper.
        
        Args:
            policy_path: Path to trained policy checkpoint
            budget: Agent's budget
            cpa: Target CPA
            category: Agent category
            name: Agent name
            state_dim: State dimension
            hidden_dim: Hidden layer size
        """
        self.budget = budget
        self.remaining_budget = budget
        self.cpa = cpa
        self.category = category
        self.name = name
        
        # Load policy
        self.policy = GaussianPolicy(state_dim, action_dim=1, hidden_dim=hidden_dim)
        if os.path.exists(policy_path):
            self.policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
            logger.info(f"Loaded PPO policy from {policy_path}")
        else:
            logger.warning(f"Policy not found at {policy_path}, using random initialization")
        
        self.policy.eval()
        
        self.tick_index = 0
        self._reset_state_tracking()
    
    def reset(self):
        """Reset budget and state tracking."""
        self.remaining_budget = self.budget
        self.tick_index = 0
        self._reset_state_tracking()
    
    def _reset_state_tracking(self):
        """Reset state tracking variables."""
        self._sum_bid = 0.0
        self._sum_lwc = 0.0
        self._sum_conv = 0.0
        self._sum_xi = 0.0
        self._sum_pvalue = 0.0
        self._count = 0
        
        self._last3_bids = []
        self._last3_lwc = []
        self._last3_conv = []
        self._last3_xi = []
        self._last3_pvalue = []
        self._tick_volumes = []
    
    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, 
                historyBid, historyAuctionResult, historyImpressionResult, 
                historyLeastWinningCost):
        """
        Generate bids using PPO policy.
        
        Args:
            timeStepIndex: Current timestep
            pValues: Conversion probabilities
            pValueSigmas: Prediction uncertainties
            historyPValueInfo: Historical pValue info
            historyBid: Historical bids
            historyAuctionResult: Historical auction results
            historyImpressionResult: Historical impression results
            historyLeastWinningCost: Historical least winning costs
            
        Returns:
            Bids for all opportunities
        """
        # Build state vector
        state = self._build_state(
            timeStepIndex, pValues, historyBid, historyAuctionResult,
            historyImpressionResult, historyLeastWinningCost
        )
        
        # Get action (alpha) from policy
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mu, std = self.policy(state_tensor)
            alpha = mu.item()  # Deterministic for evaluation
        
        alpha = np.clip(alpha, 1.0, self.cpa * 1.5)
        
        bids = alpha * pValues
        
        self._update_state_tracking(
            timeStepIndex, pValues, bids, historyAuctionResult,
            historyImpressionResult, historyLeastWinningCost
        )
        
        self.tick_index += 1
        
        return bids
    
    def _build_state(self, timeStepIndex, pValues, historyBid, 
                     historyAuctionResult, historyImpressionResult, 
                     historyLeastWinningCost):
        """Build 16-feature state vector."""
        NUM_TICK = 48
        
        # Time and budget features
        timeleft = (NUM_TICK - timeStepIndex) / NUM_TICK
        bgtleft = self.remaining_budget / self.budget if self.budget > 0 else 0.0
        
        # All-time averages
        avg_bid_all = self._sum_bid / self._count if self._count > 0 else 0.0
        avg_lwc_all = self._sum_lwc / self._count if self._count > 0 else 0.0
        avg_pvalue_all = self._sum_pvalue / self._count if self._count > 0 else 0.0
        avg_conv_all = self._sum_conv / self._count if self._count > 0 else 0.0
        avg_xi_all = self._sum_xi / self._count if self._count > 0 else 0.0
        
        # Last-3 averages
        avg_bid_last3 = np.mean(self._last3_bids) if self._last3_bids else 0.0
        avg_lwc_last3 = np.mean(self._last3_lwc) if self._last3_lwc else 0.0
        avg_pvalue_last3 = np.mean(self._last3_pvalue) if self._last3_pvalue else 0.0
        avg_conv_last3 = np.mean(self._last3_conv) if self._last3_conv else 0.0
        avg_xi_last3 = np.mean(self._last3_xi) if self._last3_xi else 0.0
        
        # Current tick features
        pvalue_mean_now = pValues.mean() if len(pValues) > 0 else 0.0
        tick_volume_now = len(pValues)
        last3_volume = sum(self._tick_volumes[-3:]) if self._tick_volumes else 0
        historical_volume = sum(self._tick_volumes)
        
        state = np.array([
            timeleft, bgtleft,
            avg_bid_all, avg_bid_last3,
            avg_lwc_all, avg_pvalue_all,
            avg_conv_all, avg_xi_all,
            avg_lwc_last3, avg_pvalue_last3,
            avg_conv_last3, avg_xi_last3,
            pvalue_mean_now, float(tick_volume_now),
            float(last3_volume), float(historical_volume),
        ], dtype=np.float32)
        
        return state
    
    def _update_state_tracking(self, timeStepIndex, pValues, bids, 
                               historyAuctionResult, historyImpressionResult, 
                               historyLeastWinningCost):
        """Update running statistics."""
        tick_bid_mean = bids.mean() if len(bids) > 0 else 0.0
        tick_pvalue_mean = pValues.mean() if len(pValues) > 0 else 0.0
        tick_volume = len(pValues)
        
        # Get least winning cost
        if len(historyLeastWinningCost) > timeStepIndex:
            tick_lwc_mean = historyLeastWinningCost[timeStepIndex].mean()
        else:
            tick_lwc_mean = 0.0
        
        # Get auction results
        if len(historyAuctionResult) > timeStepIndex:
            tick_xi_mean = historyAuctionResult[timeStepIndex][:, 0].mean()
        else:
            tick_xi_mean = 0.0
        
        # Get impression results
        if len(historyImpressionResult) > timeStepIndex:
            tick_conv_mean = historyImpressionResult[timeStepIndex][:, 1].mean()
        else:
            tick_conv_mean = 0.0
        
        # Update running totals
        self._sum_bid += tick_bid_mean
        self._sum_lwc += tick_lwc_mean
        self._sum_pvalue += tick_pvalue_mean
        self._sum_xi += tick_xi_mean
        self._sum_conv += tick_conv_mean
        self._count += 1
        
        # Update rolling windows
        self._last3_bids.append(tick_bid_mean)
        self._last3_lwc.append(tick_lwc_mean)
        self._last3_pvalue.append(tick_pvalue_mean)
        self._last3_xi.append(tick_xi_mean)
        self._last3_conv.append(tick_conv_mean)
        self._tick_volumes.append(tick_volume)
        
        # Keep only last 3
        if len(self._last3_bids) > 3:
            self._last3_bids.pop(0)
            self._last3_lwc.pop(0)
            self._last3_pvalue.pop(0)
            self._last3_xi.pop(0)
            self._last3_conv.pop(0)


def initialize_ppo_agents(
    agent0_policy_path: str,
    agent1_policy_path: str,
    agent0_budget: float = 3000,
    agent1_budget: float = 3000,
    agent0_cpa: float = 100,
    agent1_cpa: float = 100,
) -> List[PPOAgent]:
    """Initialize two PPO agents."""
    agents = [
        PPOAgent(
            policy_path=agent0_policy_path,
            budget=agent0_budget,
            cpa=agent0_cpa,
            category=0,
            name="PPO_Agent_0"
        ),
        PPOAgent(
            policy_path=agent1_policy_path,
            budget=agent1_budget,
            cpa=agent1_cpa,
            category=0,
            name="PPO_Agent_1"
        )
    ]
    return agents


def run_two_agent_ppo_test(
    agent0_policy_path: str,
    agent1_policy_path: str,
    generate_log: bool = False,
    num_episode: int = 100,
    num_tick: int = 48,
    agent0_budget: float = 3000,
    agent1_budget: float = 3000,
    agent0_cpa: float = 100,
    agent1_cpa: float = 100,
    pv_num: int = 500000,
    reserve_pv_price: float = 0.0001,
    min_remaining_budget: float = 0.1,
) -> Dict:
    """
    Run two-agent PPO test matching run_test.py structure.
    
    Args:
        agent0_policy_path: Path to agent 0's policy
        agent1_policy_path: Path to agent 1's policy
        generate_log: Whether to generate training logs
        num_episode: Number of episodes to run
        num_tick: Number of ticks per episode
        agent0_budget: Agent 0's budget
        agent1_budget: Agent 1's budget
        agent0_cpa: Agent 0's target CPA
        agent1_cpa: Agent 1's target CPA
        pv_num: Number of PV opportunities
        reserve_pv_price: Reserve price for auctions
        min_remaining_budget: Minimum budget threshold
        
    Returns:
        Dictionary with test results
    """
    # Initialize 2 PPO agents
    ppo_agents = initialize_ppo_agents(
        agent0_policy_path, agent1_policy_path,
        agent0_budget, agent1_budget,
        agent0_cpa, agent1_cpa
    )
    
    # Add 2 dummy agents to reach minimum of 4 agents (needed for BiddingEnv)
    # These agents never bid (budget = 0)
    dummy_agents = [
        PidBiddingStrategy(
            budget=0,
            cpa=100,
            category=1,
            name=f"Dummy_Agent_{i}",
            exp_tempral_ratio=np.ones(48)
        ) for i in range(2)
    ]
    
    # Combine: 2 real PPO agents + 2 dummy agents = 4 total
    agents = ppo_agents + dummy_agents
    num_agent = len(agents)  # 4
    num_real_agents = 2  # Only first 2 are real
    
    # Initialize environment components
    envs = BiddingEnv()
    envs.reserve_pv_price = reserve_pv_price
    envs.min_remaining_budget = min_remaining_budget
    
    # CRITICAL: Configure BiddingEnv for 4 agents (not default 48)
    envs.NUM_ADVERTISERS = 4
    envs.advertiser_trunc_values = [(1, 0.01)] * 4  # 4 agents
    
    # PV generator for 4 agents (but we'll only use first 2 columns)
    pv_generator = NeurIPSPvGen(
        num_tick=num_tick,
        num_agent=4,  # Generate for 4 agents
        num_agent_category=2,
        num_category=2,
        pv_num=pv_num
    )
    
    # Initialize trackers
    train_data_tracker = BiddingTracker("two_agent_ppo_tracker") if generate_log else None
    player_analyses = [
        PlayerAnalysis(f"agent{i}_analysis") 
        for i in range(num_real_agents)  # Only track real agents
    ]
    
    # Agent metadata (for all 4 agents, but only first 2 matter)
    agents_category = np.array([agent.category for agent in agents])
    agents_cpa = np.array([agent.cpa for agent in agents])
    
    logger.info("="*60)
    logger.info("TWO-AGENT PPO TEST")
    logger.info("="*60)
    for i in range(num_real_agents):
        logger.info(f"Agent {i}: {agents[i].name}, Budget=${agents[i].budget}, CPA=${agents[i].cpa}")
    logger.info(f"Episodes: {num_episode}, Ticks per episode: {num_tick}")
    logger.info("="*60)
    
    begin_time = time.time()
    
    # Results storage
    all_episode_results = []
    
    for episode in range(num_episode):
        logger.info(f"\nEpisode {episode+1}/{num_episode}")
        
        if generate_log:
            train_data_tracker.reset()
        
        # Reset episode state (only track real agents)
        rewards = np.zeros(num_agent)  # All 4 agents
        costs = np.zeros(num_agent)    # All 4 agents
        budgets = np.array([agent.budget for agent in agents])
        
        history_pvalue_infos = []
        history_bids = []
        history_auction_results = []
        history_impression_results = []
        history_least_winning_costs = []
        
        # Reset environment and agents
        pv_generator.reset(episode=episode)
        envs.reset(episode=episode)
        
        # Re-ensure correct configuration after reset (reset() may override)
        envs.NUM_ADVERTISERS = 4
        envs.advertiser_trunc_values = [(1, 0.01)] * 4
        
        for agent in agents:
            agent.reset()
        
        # Run episode
        for tick_index in range(num_tick):
            pv_values = pv_generator.pv_values[tick_index]  # (num_pv, 4)
            pvalue_sigmas = pv_generator.pValueSigmas[tick_index]  # (num_pv, 4)
            
            # Collect bids from all agents
            bids = [
                agent.bidding(
                    tick_index,
                    pv_values[:, i],
                    pvalue_sigmas[:, i],
                    [x[i] for x in history_pvalue_infos],
                    [x[i] for x in history_bids],
                    [x[i] for x in history_auction_results],
                    [x[i] for x in history_impression_results],
                    history_least_winning_costs
                ) if agent.remaining_budget >= envs.min_remaining_budget
                else np.zeros(pv_values.shape[0])
                for i, agent in enumerate(agents)
            ]
            
            bids = np.array(bids).transpose()
            bids[bids < 0] = 0
            
            remaining_budget_list = np.array([agent.remaining_budget for agent in agents])
            done_list = np.ones(len(agents), dtype=int) if tick_index == (num_tick - 1) else (
                remaining_budget_list < envs.min_remaining_budget
            ).astype(int)
            
            # Run auction with overcost guard
            ratio_max = None
            while ratio_max is None or ratio_max > 0:
                if ratio_max and ratio_max > 0:
                    over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                    adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)
                
                xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, least_winning_cost_pit, market_price_pit = \
                    envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)
                
                real_cost = cost_pit * is_exposed_pit
                cost = real_cost.sum(axis=1)
                reward = conversion_action_pit.sum(axis=1)
                
                winner_pit = get_winner(slot_pit)
                over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                ratio_max = over_cost_ratio.max()
            
            # Update agent budgets
            for i, agent in enumerate(agents):
                agent.remaining_budget -= cost[i]
            
            rewards += reward
            costs += cost
            
            # Update history
            history_bids.append(bids.transpose())
            history_least_winning_costs.append(least_winning_cost_pit)
            pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
            history_pvalue_infos.append(pvalue_info)
            auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
            history_auction_results.append(auction_info)
            impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
            history_impression_results.append(impression_info)
            
            # Log data
            if generate_log:
                train_data_tracker.train_logging(
                    episode, tick_index, pv_values, budgets, agents_cpa, agents_category,
                    remaining_budget_list, 0, pvalue_sigmas, bids,
                    xi_pit, slot_pit, cost_pit, is_exposed_pit,
                    conversion_action_pit, least_winning_cost_pit, done_list
                )
            
            # Player analysis (only for real agents)
            for player_index in range(num_real_agents):
                tick_win_pv = np.sum(is_exposed_pit[player_index])
                tick_compete_pv = len(xi_pit[player_index])
                tick_all_win_bid = np.sum(bids[:, player_index] * is_exposed_pit[player_index])
                bid_mean = np.mean(bids[:, player_index])
                player_analyses[player_index].logging_player_tick(
                    episode, tick_index, player_index, agents_cpa[player_index],
                    budgets[player_index], reward[player_index],
                    cost[player_index], tick_compete_pv, tick_win_pv,
                    tick_all_win_bid, bid_mean
                )
        
        # Generate log if requested
        if generate_log:
            os.makedirs("data/log", exist_ok=True)
            train_data_tracker.generate_train_data(f"data/log/two_agent_ep{episode}.csv")
        
        # Store episode results (only for real agents)
        episode_result = {
            'episode': episode,
            'rewards': rewards[:num_real_agents].copy(),
            'costs': costs[:num_real_agents].copy(),
            'cpa': [costs[i] / rewards[i] if rewards[i] > 0 else 0 for i in range(num_real_agents)],
            'budget_used': [budgets[i] - agents[i].remaining_budget for i in range(num_real_agents)],
            'budget_utilization': [(budgets[i] - agents[i].remaining_budget) / budgets[i] 
                                  for i in range(num_real_agents)],
        }
        all_episode_results.append(episode_result)
        
        # Log episode summary
        logger.info(f"  Agent 0: Conv={rewards[0]:.1f}, Cost=${costs[0]:.2f}, CPA=${episode_result['cpa'][0]:.2f}")
        logger.info(f"  Agent 1: Conv={rewards[1]:.1f}, Cost=${costs[1]:.2f}, CPA=${episode_result['cpa'][1]:.2f}")
    
    end_time = time.time()
    logger.info(f"\nTotal time elapsed: {end_time - begin_time:.2f} seconds")
    
    # Aggregate results (only for real agents)
    results = {
        'agents': [agents[i].name for i in range(num_real_agents)],
        'episodes': all_episode_results,
        'summary': {}
    }
    
    for i in range(num_real_agents):
        player_analyses[i].player_multi_episode(agents[i].name)
        agent_result = player_analyses[i].get_return_res(
            agents[i].name, i, agents[i].category
        )
        results['summary'][f'agent{i}'] = agent_result
    
    # Print summary (only real agents)
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for i in range(num_real_agents):
        rewards_list = [ep['rewards'][i] for ep in all_episode_results]
        costs_list = [ep['costs'][i] for ep in all_episode_results]
        cpa_list = [ep['cpa'][i] for ep in all_episode_results]
        
        logger.info(f"\nAgent {i} ({agents[i].name}):")
        logger.info(f"  Avg Conversions: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
        logger.info(f"  Avg Cost:        ${np.mean(costs_list):.2f} ± ${np.std(costs_list):.2f}")
        logger.info(f"  Avg CPA:         ${np.mean(cpa_list):.2f} ± ${np.std(cpa_list):.2f}")
        logger.info(f"  Win Rate:        {np.mean([1 if ep['rewards'][i] > ep['rewards'][1-i] else 0 for ep in all_episode_results])*100:.1f}%")
    
    return results


if __name__ == "__main__":
    # Configuration matching test.gin
    SAVE_PATH = "strategy_train_env/saved_model/two_agent_ppo"
    
    # Parameters from test.gin
    RESERVE_PV_PRICE = 0.0001
    MIN_REMAINING_BUDGET = 0.1
    PVNUM = 500000
    NUM_EPISODE = 100  # More episodes for testing
    NUM_TICK = 48
    
    results = run_two_agent_ppo_test(
        agent0_policy_path=f"{SAVE_PATH}/agent0_policy_final.pt",
        agent1_policy_path=f"{SAVE_PATH}/agent1_policy_final.pt",
        generate_log=False,
        num_episode=NUM_EPISODE,
        num_tick=NUM_TICK,
        agent0_budget=3000,
        agent1_budget=3000,
        agent0_cpa=100,
        agent1_cpa=100,
        pv_num=PVNUM,
        reserve_pv_price=RESERVE_PV_PRICE,
        min_remaining_budget=MIN_REMAINING_BUDGET,
    )