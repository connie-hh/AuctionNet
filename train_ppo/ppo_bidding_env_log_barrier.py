import logging
import numpy as np
from typing import Tuple, Optional
from run.run_test import get_winner, adjust_over_cost
from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
eps = 1e-8
tau = 0.01

class PpoBiddingEnv_LogBarrier:
    """
    Finalized Gym-style environment for PPO. 
    Implements Episodic Reward logic to avoid the 'Reward Design Trap'[cite: 170].
    """

    NUM_TICK = 48
    STATE_DIM = 17  # Incremented to include BCR_t 

    def __init__(self, player_index: int = 0, episode: int = 0):
        self.player_index = player_index
        self.episode = episode
        dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
        self.bidding_controller = Controller(
            player_index=player_index,
            player_agent=dummy_agent,
            num_tick=48,
            num_agent_category=8,
            num_category=6,
            pv_num=1000,
        )
        self.agents = self.bidding_controller.agents
        self.envs = self.bidding_controller.biddingEnv
        self.pv_generator = self.bidding_controller.pvGenerator
        self.num_agent = len(self.agents)
        self._reset_episode_state()

    def reset(self, episode: Optional[int] = None) -> np.ndarray:
        if episode is not None:
            self.episode = episode
        else:
            self.episode += 1
        self.bidding_controller.reset(episode=self.episode)
        self._reset_episode_state()
        return self._build_state()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        tick = self.tick_index
        pv_values = self.pv_generator.pv_values[tick]
        pvalue_sigmas = self.pv_generator.pValueSigmas[tick]

        # Action: Scalar alpha. Bid = alpha * value [cite: 99, 161]
        bids = self._collect_bids(pv_values, pvalue_sigmas, action, tick)
        bids = np.array(bids).T
        bids[bids < 0] = 0
        
        # Capture budget BEFORE tick for BCR calculation
        budget_before = self.agents[self.player_index].remaining_budget

        # Hard budget enforcement loop (Reactive Clipping) [cite: 144]
        remaining_budgets = np.array([a.remaining_budget for a in self.agents])
        bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit = \
            self._run_auction_with_overcost_guard(pv_values, pvalue_sigmas, bids, remaining_budgets)

        # Update budgets
        real_cost = (cost_pit * is_exposed_pit)
        cost_per_agent = real_cost.sum(axis=1)
        new_remaining = remaining_budgets - cost_per_agent
        
        reward_per_agent = (
            conversion_action_pit.sum(axis=1)
            + tau * np.log(new_remaining + eps)
        )

        for i, agent in enumerate(self.agents):
            agent.remaining_budget -= cost_per_agent[i]

        self._update_history(pv_values, pvalue_sigmas, bids, xi_pit, slot_pit, cost_pit, 
                             is_exposed_pit, conversion_action_pit, lwc_pit)

        player_cost = cost_per_agent[self.player_index]
        player_reward = float(reward_per_agent[self.player_index])
        player_budget_after = self.agents[self.player_index].remaining_budget
        
        self.cumulative_cost += player_cost
        self.cumulative_reward += player_reward
        self.tick_index += 1

        # Budget Consumption Rate (BCR) 
        self.last_bcr = (budget_before - player_budget_after) / (budget_before + 1e-4)

        done = (self.tick_index >= self.NUM_TICK or player_budget_after < self.envs.min_remaining_budget)

        # --- REWARD SHAPING: Episodic Reward [cite: 182, 186] ---
        # We provide 0 reward during the episode and return total return at the end.
        # This prevents the agent from being 'obsessed' with immediate tick wins[cite: 171].
        ppo_reward = self.cumulative_reward if done else 0.0

        next_state = self._build_state() if not done else np.zeros(self.STATE_DIM)

        info = {
            "tick": tick,
            "cost": player_cost,
            "remaining_budget": player_budget_after,
            "cumulative_reward": self.cumulative_reward,
            "bcr": self.last_bcr
        }

        return next_state, ppo_reward, done, info

    def _reset_episode_state(self) -> None:
        self.tick_index = 0
        self.cumulative_cost = 0.0
        self.cumulative_reward = 0.0
        self.last_bcr = 0.0
        self.history_pvalue_infos, self.history_bids = [], []
        self.history_auction_results, self.history_impression_results = [], []
        self.history_least_winning_costs = []
        self._sum_bid, self._sum_lwc, self._sum_conv, self._sum_xi, self._sum_pvalue = [np.zeros(self.num_agent) for _ in range(5)]
        self._count = np.zeros(self.num_agent)
        self._last3_bids, self._last3_lwc, self._last3_conv, self._last3_xi, self._last3_pvalue, self._tick_volumes = [[] for _ in range(6)]

    def _collect_bids(self, pv_values, pvalue_sigmas, player_action, tick):
        bids = []
        for i, agent in enumerate(self.agents):
            if agent.remaining_budget < self.envs.min_remaining_budget:
                bids.append(np.zeros(pv_values.shape[0]))
            elif i == self.player_index:
                bids.append(player_action * pv_values[:, i]) # Optimal linear bid form [cite: 99]
            else:
                bids.append(agent.bidding(tick, pv_values[:, i], pvalue_sigmas[:, i], 
                                          [x[i] for x in self.history_pvalue_infos],
                                          [x[i] for x in self.history_bids],
                                          [x[i] for x in self.history_auction_results],
                                          [x[i] for x in self.history_impression_results],
                                          self.history_least_winning_costs))
        return bids

    def _run_auction_with_overcost_guard(self, pv_values, pvalue_sigmas, bids, remaining_budgets):
        ratio_max = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                real_cost = (cost_pit * is_exposed_pit).sum(axis=1)
                over_cost_ratio = np.maximum((real_cost - remaining_budgets) / (real_cost + 1e-4), 0)
                winner_pit = get_winner(slot_pit)
                adjust_over_cost(bids, over_cost_ratio, self.envs.slot_coefficients, winner_pit)
            xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit, _ = self.envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)
            real_cost = (cost_pit * is_exposed_pit).sum(axis=1)
            over_cost_ratio = np.maximum((real_cost - remaining_budgets) / (real_cost + 1e-4), 0)
            ratio_max = over_cost_ratio.max()
        return bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit

    def _update_history(self, pv_values, pvalue_sigmas, bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit):
        p = self.player_index
        stats = [bids[:, p].mean(), lwc_pit.mean(), conversion_action_pit[p].mean(), xi_pit[p].mean(), pv_values[:, p].mean()]
        accumulators = [self._sum_bid, self._sum_lwc, self._sum_conv, self._sum_xi, self._sum_pvalue]
        windows = [self._last3_bids, self._last3_lwc, self._last3_conv, self._last3_xi, self._last3_pvalue]
        for i, val in enumerate(stats):
            accumulators[i][p] += val
            windows[i].append(val)
            if len(windows[i]) > 3: windows[i].pop(0)
        self._count[p] += 1
        self._tick_volumes.append(pv_values.shape[0])
        self.history_bids.append(bids.T)
        self.history_least_winning_costs.append(lwc_pit)
        self.history_pvalue_infos.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        self.history_auction_results.append(np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        self.history_impression_results.append(np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

    def _build_state(self) -> np.ndarray:
        p, tick, n = self.player_index, self.tick_index, self._count[self.player_index]
        budget, remaining = self.agents[p].budget, self.agents[p].remaining_budget
        timeleft, bgtleft = (self.NUM_TICK - tick) / self.NUM_TICK, remaining / budget if budget > 0 else 0.0
        
        # Base averages
        avg_all = [self._sum_bid[p], self._sum_lwc[p], self._sum_pvalue[p], self._sum_conv[p], self._sum_xi[p]]
        avg_all = [x / n if n > 0 else 0.0 for x in avg_all]
        avg_l3 = [np.mean(w) if w else 0.0 for w in [self._last3_bids, self._last3_lwc, self._last3_pvalue, self._last3_conv, self._last3_xi]]

        state = np.array([
            timeleft, bgtleft, self.last_bcr, # Added BCR 
            avg_all[0], avg_l3[0], avg_all[1], avg_all[2], avg_all[3], avg_all[4],
            avg_l3[1], avg_l3[2], avg_l3[3], avg_l3[4],
            self.pv_generator.pv_values[tick][:, p].mean() if tick < self.NUM_TICK else 0.0,
            float(len(self.pv_generator.pv_values[tick][:, p])) if tick < self.NUM_TICK else 0.0,
            float(sum(self._tick_volumes[-3:])), float(sum(self._tick_volumes))
        ], dtype=np.float32)
        return state