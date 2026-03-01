import logging
import numpy as np
from typing import Tuple, Optional, List
from run.run_test import get_winner, adjust_over_cost
from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PpoBiddingEnv:
    """
    Gym-style environment wrapper for PPO training.
    Modified to train against a specific subset of agents identified by multiple tags.
    """

    NUM_TICK = 48
    STATE_DIM = 16

    def __init__(self, player_index: int = 0, episode: int = 0, competitor_subset_tags: List[str] = ["IQL"]):
        self.player_index = player_index
        self.episode = episode
        # Ensure tags are stored as a list for the 'any' check
        self.competitor_subset_tags = competitor_subset_tags if isinstance(competitor_subset_tags, list) else [competitor_subset_tags]

        # Initialise controller with dummy agent (PPO overrides its actions)
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

        # Identify the subset of active agents using the multi-tag logic
        self.active_agent_indices = self._identify_subset_indices()
        
        active_names = [self.agents[i].name for i in self.active_agent_indices if i != self.player_index]
        logger.info(f"PpoBiddingEnv initialized. Tags: {self.competitor_subset_tags}")
        logger.info(f"Active competitors ({len(active_names)}): {active_names}")

        self.num_agent = len(self.agents)
        self._reset_episode_state()

    def _identify_subset_indices(self) -> List[int]:
        """Identifies indices of agents matching ANY of the tags, plus the player index."""
        indices = [self.player_index]
        for i, agent in enumerate(self.agents):
            if i == self.player_index:
                continue
            # Multi-tag check: matches if ANY tag in the list is found in the agent name
            if any(tag in agent.name for tag in self.competitor_subset_tags):
                indices.append(i)
        return indices

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

        # Only subset participates
        bids = self._collect_bids(pv_values, pvalue_sigmas, action, tick)
        bids = np.array(bids).T
        bids[bids < 0] = 0

        remaining_budgets = np.array([a.remaining_budget for a in self.agents])
        bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit = \
            self._run_auction_with_overcost_guard(pv_values, pvalue_sigmas, bids, remaining_budgets)

        real_cost = (cost_pit * is_exposed_pit)
        cost_per_agent = real_cost.sum(axis=1)
        reward_per_agent = conversion_action_pit.sum(axis=1)

        for i, agent in enumerate(self.agents):
            agent.remaining_budget -= cost_per_agent[i]

        self._update_history(pv_values, pvalue_sigmas, bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit)

        player_cost = cost_per_agent[self.player_index]
        player_reward = float(reward_per_agent[self.player_index])
        player_budget = self.agents[self.player_index].remaining_budget

        self.cumulative_cost += player_cost
        self.cumulative_reward += player_reward
        self.tick_index += 1

        done = (self.tick_index >= self.NUM_TICK or player_budget < self.envs.min_remaining_budget)
        next_state = self._build_state() if not done else np.zeros(self.STATE_DIM)

        info = {
            "tick": tick, "cost": player_cost, "remaining_budget": player_budget,
            "cumulative_reward": self.cumulative_reward, "cumulative_cost": self.cumulative_cost,
            "least_winning_cost": lwc_pit.mean(),
        }
        return next_state, player_reward, done, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_episode_state(self) -> None:
        self.tick_index = 0
        self.cumulative_cost = 0.0
        self.cumulative_reward = 0.0
        self.history_pvalue_infos, self.history_bids = [], []
        self.history_auction_results, self.history_impression_results = [], []
        self.history_least_winning_costs = []
        self._sum_bid, self._sum_lwc = np.zeros(self.num_agent), np.zeros(self.num_agent)
        self._sum_conv, self._sum_xi = np.zeros(self.num_agent), np.zeros(self.num_agent)
        self._sum_pvalue, self._count = np.zeros(self.num_agent), np.zeros(self.num_agent)
        self._last3_bids, self._last3_lwc, self._last3_conv = [], [], []
        self._last3_xi, self._last3_pvalue, self._tick_volumes = [], [], []

    def _collect_bids(self, pv_values, pvalue_sigmas, player_action, tick) -> list:
        bids = []
        for i, agent in enumerate(self.agents):
            if i not in self.active_agent_indices:
                bids.append(np.zeros(pv_values.shape[0]))
                continue
            if agent.remaining_budget < self.envs.min_remaining_budget:
                bids.append(np.zeros(pv_values.shape[0]))
            elif i == self.player_index:
                bids.append(player_action * pv_values[:, i])
            else:
                bids.append(agent.bidding(tick, pv_values[:, i], pvalue_sigmas[:, i], 
                    [x[i] for x in self.history_pvalue_infos], [x[i] for x in self.history_bids],
                    [x[i] for x in self.history_auction_results], [x[i] for x in self.history_impression_results],
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
            ratio_max = np.maximum((real_cost - remaining_budgets) / (real_cost + 1e-4), 0).max()
        return bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit

    def _update_history(self, pv_values, pvalue_sigmas, bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit) -> None:
        p = self.player_index
        tick_pv_num = pv_values.shape[0]
        m_bid, m_lwc, m_conv = bids[:, p].mean(), lwc_pit.mean(), conversion_action_pit[p].mean()
        m_xi, m_pv = xi_pit[p].mean(), pv_values[:, p].mean()
        self._sum_bid[p], self._sum_lwc[p], self._sum_conv[p] = self._sum_bid[p]+m_bid, self._sum_lwc[p]+m_lwc, self._sum_conv[p]+m_conv
        self._sum_xi[p], self._sum_pvalue[p], self._count[p] = self._sum_xi[p]+m_xi, self._sum_pvalue[p]+m_pv, self._count[p]+1
        self._last3_bids.append(m_bid); self._last3_lwc.append(m_lwc); self._last3_conv.append(m_conv)
        self._last3_xi.append(m_xi); self._last3_pvalue.append(m_pv); self._tick_volumes.append(tick_pv_num)
        if len(self._last3_bids) > 3:
            for l in [self._last3_bids, self._last3_lwc, self._last3_conv, self._last3_xi, self._last3_pvalue]: l.pop(0)
        self.history_bids.append(bids.T); self.history_least_winning_costs.append(lwc_pit)
        self.history_pvalue_infos.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        self.history_auction_results.append(np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        self.history_impression_results.append(np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

    def _build_state(self) -> np.ndarray:
        p, tick, n = self.player_index, self.tick_index, self._count[self.player_index]
        bgt_ratio = self.agents[p].remaining_budget / self.agents[p].budget if self.agents[p].budget > 0 else 0.0
        cur_pv = self.pv_generator.pv_values[tick][:, p] if tick < self.NUM_TICK else np.array([0])
        state = np.array([
            (self.NUM_TICK - tick) / self.NUM_TICK, bgt_ratio,
            self._sum_bid[p]/n if n>0 else 0, np.mean(self._last3_bids) if self._last3_bids else 0,
            self._sum_lwc[p]/n if n>0 else 0, self._sum_pvalue[p]/n if n>0 else 0,
            self._sum_conv[p]/n if n>0 else 0, self._sum_xi[p]/n if n>0 else 0,
            np.mean(self._last3_lwc) if self._last3_lwc else 0, np.mean(self._last3_pvalue) if self._last3_pvalue else 0,
            np.mean(self._last3_conv) if self._last3_conv else 0, np.mean(self._last3_xi) if self._last3_xi else 0,
            cur_pv.mean(), float(len(cur_pv)), float(sum(self._tick_volumes[-3:])), float(sum(self._tick_volumes))
        ], dtype=np.float32)
        return state