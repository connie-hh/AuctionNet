import logging
import numpy as np
from typing import Tuple, Optional, List
from run.run_test import get_winner, adjust_over_cost
from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoAgentPpoBiddingEnv:
    """
    Two-agent head-to-head PPO training environment.
    
    Two PPO agents compete against each other (no other agents).
    
    State vector (16 features per agent):
        timeleft, bgtleft,
        avg_bid_all, avg_bid_last_3,
        avg_leastWinningCost_all, avg_pValue_all,
        avg_conversionAction_all, avg_xi_all,
        avg_leastWinningCost_last_3, avg_pValue_last_3,
        avg_conversionAction_last_3, avg_xi_last_3,
        pValue_mean, timeStepIndex_volume,
        last_3_timeStepIndexs_volume, historical_volume

    Actions:
        Two scalar alphas (CPA multipliers). Bids are computed as alpha * pValues.
    """

    NUM_TICK = 48
    STATE_DIM = 16

    def __init__(self, episode: int = 0):
        """
        Initialize environment with 2 agents.
        
        Args:
            episode: Starting episode number
        """
        self.episode = episode
        
        # Create 2 agents manually using dummy strategy objects
        # (We only need their budget/cpa/category attributes, not their bidding logic)
        agent0 = PidBiddingStrategy(
            budget=2900,
            cpa=100,
            category=0,
            name="PPO_Agent_0",
            exp_tempral_ratio=np.ones(48)
        )
        agent1 = PidBiddingStrategy(
            budget=4350,
            cpa=70,
            category=0,
            name="PPO_Agent_1",
            exp_tempral_ratio=np.ones(48)
        )
        
        self.agents = [agent0, agent1]
        self.num_agent = 2
        
        # Create environment - configure for 4 agents (2 real + 2 padding)
        self.envs = BiddingEnv()
        self.envs.NUM_ADVERTISERS = 4  # Override default of 48
        self.envs.advertiser_trunc_values = [(1, 0.01)] * 4  # Initialize for 4 agents
        
        self.pv_generator = NeurIPSPvGen(
            episode=episode,
            num_tick=48,
            num_agent=2,
            num_agent_category=2,
            num_category=1,
            pv_num=500000
        )
        
        # Initialize episode state
        self._reset_episode_state()
        
        logger.info(f"Initialized 2-agent environment:")
        logger.info(f"  Agent 0: Budget=${self.agents[0].budget}, CPA=${self.agents[0].cpa}")
        logger.info(f"  Agent 1: Budget=${self.agents[1].budget}, CPA=${self.agents[1].cpa}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, episode: Optional[int] = None) -> List[np.ndarray]:
        """
        Reset environment for a new episode.

        Args:
            episode: Episode index. If None, increments the current episode.

        Returns:
            List of initial state vectors [state_agent0, state_agent1]
        """
        if episode is not None:
            self.episode = episode
        else:
            self.episode += 1

        # Reset PV generator for new episode
        self.pv_generator.reset(episode=self.episode)
        
        # Reset BiddingEnv for new episode (initializes advertiser_trunc_values)
        self.envs.reset(episode=self.episode)
        
        # Reset both agents
        for agent in self.agents:
            agent.reset()
        
        # Reset episode state
        self._reset_episode_state()
        
        # Build initial states for both agents
        state0 = self._build_state(agent_idx=0)
        state1 = self._build_state(agent_idx=1)
        
        return [state0, state1]

    def step(self, actions: List[float]) -> Tuple[List[np.ndarray], List[float], List[bool], List[dict]]:
        """
        Advance environment by one timestep.

        Args:
            actions: [alpha_agent0, alpha_agent1] - CPA multipliers for both agents

        Returns:
            next_states: [state_agent0, state_agent1]
            rewards: [reward_agent0, reward_agent1] - conversions won
            dones: [done_agent0, done_agent1]
            infos: [info_agent0, info_agent1] - diagnostic dicts
        """
        tick = self.tick_index
        pv_values_original = self.pv_generator.pv_values[tick]        # (num_pv, 2) - keep for logging
        pvalue_sigmas_original = self.pv_generator.pValueSigmas[tick] # (num_pv, 2) - keep for logging
        
        pv_values = pv_values_original.copy()
        pvalue_sigmas = pvalue_sigmas_original.copy()

        # Pad pv_values and pvalue_sigmas to 4 agents for BiddingEnv compatibility
        num_pv = pv_values.shape[0]
        if pv_values.shape[1] < 4:
            pv_padding = np.zeros((num_pv, 4 - pv_values.shape[1]))
            pv_values = np.hstack([pv_values, pv_padding])  # Shape: (num_pv, 4)
            pvalue_sigmas = np.hstack([pvalue_sigmas, pv_padding])  # Shape: (num_pv, 4)

        # Build bids for both agents (using original pv_values columns)
        bids = np.array([
            actions[0] * pv_values_original[:, 0],  # Agent 0's bids
            actions[1] * pv_values_original[:, 1]   # Agent 1's bids
        ]).T  # Shape: (num_pv, 2)
        
        bids[bids < 0] = 0
        
        # Pad bids to at least 4 agents for BiddingEnv compatibility
        # (BiddingEnv expects enough agents to fill 3 slots + reserve)
        if bids.shape[1] < 4:
            padding = np.zeros((num_pv, 4 - bids.shape[1]))
            bids = np.hstack([bids, padding])  # Shape: (num_pv, 4)
        
        # Diagnostic logging (first tick of first few episodes)
        if tick == 0 and self.episode < 3:
            logger.info(f"[Episode {self.episode}, Tick {tick}] Diagnostics:")
            for i in range(2):
                logger.info(f"  Agent {i}:")
                logger.info(f"    Action (alpha): {actions[i]:.4f}")
                logger.info(f"    PValues: min={pv_values_original[:, i].min():.6f}, mean={pv_values_original[:, i].mean():.6f}, max={pv_values_original[:, i].max():.6f}")
                logger.info(f"    Bids: min={bids[:, i].min():.6f}, mean={bids[:, i].mean():.6f}, max={bids[:, i].max():.6f}")

        # Run auction with overcost guard
        remaining_budgets = np.array([a.remaining_budget for a in self.agents])
        
        # Pad remaining_budgets to match padded bids
        if len(remaining_budgets) < 4:
            remaining_budgets_padded = np.zeros(4)
            remaining_budgets_padded[:2] = remaining_budgets
            remaining_budgets = remaining_budgets_padded
        
        bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit = \
            self._run_auction_with_overcost_guard(pv_values, pvalue_sigmas, bids, remaining_budgets)

        # Extract only the 2 real agents' results (ignore padding)
        xi_pit = xi_pit[:2]
        slot_pit = slot_pit[:2]
        cost_pit = cost_pit[:2]
        is_exposed_pit = is_exposed_pit[:2]
        conversion_action_pit = conversion_action_pit[:2]

        # Compute costs and rewards per agent
        real_cost = (cost_pit * is_exposed_pit)          # (2, num_pv)
        cost_per_agent = real_cost.sum(axis=1)           # (2,)
        reward_per_agent = conversion_action_pit.sum(axis=1)  # (2,)

        # Update budgets
        for i, agent in enumerate(self.agents):
            agent.remaining_budget -= cost_per_agent[i]

        # Update history
        self._update_history(
            pv_values_original, pvalue_sigmas_original, bids[:, :2],  # Use original unpadded values
            xi_pit, slot_pit, cost_pit,
            is_exposed_pit, conversion_action_pit, lwc_pit
        )

        # More diagnostic logging
        if tick == 0 and self.episode < 3:
            for i in range(2):
                logger.info(f"  Agent {i} results:")
                logger.info(f"    Wins (xi=1): {xi_pit[i].sum()}/{len(xi_pit[i])}")
                logger.info(f"    Exposures: {is_exposed_pit[i].sum()}")
                logger.info(f"    Conversions: {reward_per_agent[i]:.2f}")
                logger.info(f"    Cost: {cost_per_agent[i]:.2f}")
                logger.info(f"    Remaining budget: {self.agents[i].remaining_budget:.2f}")

        # Increment tick
        self.tick_index += 1

        # Build return values for each agent
        next_states = []
        rewards = []
        dones = []
        infos = []

        for i in range(2):
            # Check if done
            done = (
                self.tick_index >= self.NUM_TICK
                or self.agents[i].remaining_budget < self.envs.min_remaining_budget
            )
            
            # Build next state
            next_state = self._build_state(agent_idx=i) if not done else np.zeros(self.STATE_DIM)
            
            # Build info dict
            info = {
                "tick": tick,
                "cost": cost_per_agent[i],
                "remaining_budget": self.agents[i].remaining_budget,
                "conversions": reward_per_agent[i],
                "least_winning_cost": lwc_pit.mean(),
            }
            
            next_states.append(next_state)
            rewards.append(float(reward_per_agent[i]))
            dones.append(done)
            infos.append(info)

        return next_states, rewards, dones, infos

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_episode_state(self) -> None:
        """Clear per-episode tracking variables for both agents."""
        self.tick_index = 0

        # History lists (shared across both agents for compatibility)
        self.history_pvalue_infos = []
        self.history_bids = []
        self.history_auction_results = []
        self.history_impression_results = []
        self.history_least_winning_costs = []

        # Running stats per agent (use dicts keyed by agent index)
        self._sum_bid = {0: 0.0, 1: 0.0}
        self._sum_lwc = {0: 0.0, 1: 0.0}
        self._sum_conv = {0: 0.0, 1: 0.0}
        self._sum_xi = {0: 0.0, 1: 0.0}
        self._sum_pvalue = {0: 0.0, 1: 0.0}
        self._count = {0: 0, 1: 0}

        # Rolling window stats per agent
        self._last3_bids = {0: [], 1: []}
        self._last3_lwc = {0: [], 1: []}
        self._last3_conv = {0: [], 1: []}
        self._last3_xi = {0: [], 1: []}
        self._last3_pvalue = {0: [], 1: []}
        self._tick_volumes = {0: [], 1: []}

    def _run_auction_with_overcost_guard(
        self,
        pv_values: np.ndarray,
        pvalue_sigmas: np.ndarray,
        bids: np.ndarray,
        remaining_budgets: np.ndarray,
    ):
        """
        Run auction with overcost adjustment loop.
        Iteratively drops bids for over-budget agents until no agent overspends.
        """
        ratio_max = None
        xi_pit = slot_pit = cost_pit = is_exposed_pit = None
        conversion_action_pit = lwc_pit = None

        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                real_cost = (cost_pit * is_exposed_pit).sum(axis=1)
                over_cost_ratio = np.maximum(
                    (real_cost - remaining_budgets) / (real_cost + 1e-4), 0
                )
                winner_pit = get_winner(slot_pit)
                adjust_over_cost(bids, over_cost_ratio, self.envs.slot_coefficients, winner_pit)

            xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit, _ = \
                self.envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

            real_cost = (cost_pit * is_exposed_pit).sum(axis=1)
            over_cost_ratio = np.maximum(
                (real_cost - remaining_budgets) / (real_cost + 1e-4), 0
            )
            ratio_max = over_cost_ratio.max()

        return bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit

    def _update_history(
        self,
        pv_values, pvalue_sigmas, bids,
        xi_pit, slot_pit, cost_pit,
        is_exposed_pit, conversion_action_pit, lwc_pit,
    ) -> None:
        """Update all history arrays and running stat accumulators for both agents."""
        tick_pv_num = pv_values.shape[0]

        # Update stats for both agents
        for agent_idx in range(2):
            # Running totals for all-time averages
            tick_bid_mean = bids[:, agent_idx].mean()
            tick_lwc_mean = lwc_pit.mean()
            tick_conv_mean = conversion_action_pit[agent_idx].mean()
            tick_xi_mean = xi_pit[agent_idx].mean()
            tick_pvalue_mean = pv_values[:, agent_idx].mean()

            self._sum_bid[agent_idx] += tick_bid_mean
            self._sum_lwc[agent_idx] += tick_lwc_mean
            self._sum_conv[agent_idx] += tick_conv_mean
            self._sum_xi[agent_idx] += tick_xi_mean
            self._sum_pvalue[agent_idx] += tick_pvalue_mean
            self._count[agent_idx] += 1

            # Rolling window (last 3 ticks)
            self._last3_bids[agent_idx].append(tick_bid_mean)
            self._last3_lwc[agent_idx].append(tick_lwc_mean)
            self._last3_conv[agent_idx].append(tick_conv_mean)
            self._last3_xi[agent_idx].append(tick_xi_mean)
            self._last3_pvalue[agent_idx].append(tick_pvalue_mean)
            self._tick_volumes[agent_idx].append(tick_pv_num)

            # Keep only last 3
            if len(self._last3_bids[agent_idx]) > 3:
                self._last3_bids[agent_idx].pop(0)
                self._last3_lwc[agent_idx].pop(0)
                self._last3_conv[agent_idx].pop(0)
                self._last3_xi[agent_idx].pop(0)
                self._last3_pvalue[agent_idx].pop(0)

        # Standard history arrays (all agents - for compatibility)
        self.history_bids.append(bids.T)
        self.history_least_winning_costs.append(lwc_pit)
        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
        self.history_pvalue_infos.append(pvalue_info)
        auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
        self.history_auction_results.append(auction_info)
        impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
        self.history_impression_results.append(impression_info)

    def _build_state(self, agent_idx: int) -> np.ndarray:
        """
        Construct 16-feature state vector for a specific agent.
        
        Args:
            agent_idx: Which agent (0 or 1)
            
        Returns:
            State vector of shape (16,)
        """
        tick = self.tick_index
        n = self._count[agent_idx]

        budget = self.agents[agent_idx].budget
        remaining = self.agents[agent_idx].remaining_budget
        timeleft = (self.NUM_TICK - tick) / self.NUM_TICK
        bgtleft = remaining / budget if budget > 0 else 0.0

        # All-time averages (from previous ticks)
        avg_bid_all = self._sum_bid[agent_idx] / n if n > 0 else 0.0
        avg_lwc_all = self._sum_lwc[agent_idx] / n if n > 0 else 0.0
        avg_pvalue_all = self._sum_pvalue[agent_idx] / n if n > 0 else 0.0
        avg_conv_all = self._sum_conv[agent_idx] / n if n > 0 else 0.0
        avg_xi_all = self._sum_xi[agent_idx] / n if n > 0 else 0.0

        # Last-3 averages
        avg_bid_last3 = np.mean(self._last3_bids[agent_idx]) if self._last3_bids[agent_idx] else 0.0
        avg_lwc_last3 = np.mean(self._last3_lwc[agent_idx]) if self._last3_lwc[agent_idx] else 0.0
        avg_pvalue_last3 = np.mean(self._last3_pvalue[agent_idx]) if self._last3_pvalue[agent_idx] else 0.0
        avg_conv_last3 = np.mean(self._last3_conv[agent_idx]) if self._last3_conv[agent_idx] else 0.0
        avg_xi_last3 = np.mean(self._last3_xi[agent_idx]) if self._last3_xi[agent_idx] else 0.0

        # Current tick traffic features
        if tick < self.NUM_TICK:
            pv_now = self.pv_generator.pv_values[tick][:, agent_idx]
            pvalue_mean_now = pv_now.mean()
            tick_volume_now = len(pv_now)
        else:
            pvalue_mean_now = 0.0
            tick_volume_now = 0

        last3_volume = sum(self._tick_volumes[agent_idx][-3:]) if self._tick_volumes[agent_idx] else 0
        historical_volume = sum(self._tick_volumes[agent_idx])

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


# Test the environment
if __name__ == "__main__":
    env = TwoAgentPpoBiddingEnv()
    
    print("\nTesting environment...")
    states = env.reset()
    print(f"Reset returned {len(states)} states with shapes: {states[0].shape}, {states[1].shape}")
    
    # Test one step with different actions
    actions = [100.0, 200.0]  # Agent 0 conservative, Agent 1 aggressive
    next_states, rewards, dones, infos = env.step(actions)
    
    print(f"\nAfter one step:")
    print(f"  Agent 0: Reward={rewards[0]:.2f}, Cost={infos[0]['cost']:.2f}, Budget={infos[0]['remaining_budget']:.2f}")
    print(f"  Agent 1: Reward={rewards[1]:.2f}, Cost={infos[1]['cost']:.2f}, Budget={infos[1]['remaining_budget']:.2f}")
    print(f"  Dones: {dones}")