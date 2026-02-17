import numpy as np
from typing import Tuple, Optional
from run.run_test import get_winner, adjust_over_cost
from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy


class PpoBiddingEnv:
    """
    Gym-style environment wrapper for PPO training, adapted from run_test.py.

    Wraps the existing BiddingEnv auction simulator and NeurIPSPvGen traffic
    generator into a step-by-step interface that PPO can interact with.

    State vector (16 features), matching TrainDataGenerator:
        timeleft, bgtleft,
        avg_bid_all, avg_bid_last_3,
        avg_leastWinningCost_all, avg_pValue_all,
        avg_conversionAction_all, avg_xi_all,
        avg_leastWinningCost_last_3, avg_pValue_last_3,
        avg_conversionAction_last_3, avg_xi_last_3,
        pValue_mean, timeStepIndex_volume,
        last_3_timeStepIndexs_volume, historical_volume

    Action:
        A scalar alpha (CPA multiplier). Bids are computed as alpha * pValues.
    """

    NUM_TICK = 48
    STATE_DIM = 16

    def __init__(self, player_index: int = 0, episode: int = 0):
        self.player_index = player_index
        self.episode = episode

        # Initialise controller with dummy agent (PPO overrides its actions)
        dummy_agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
        self.bidding_controller = Controller(
            player_index=player_index,
            player_agent=dummy_agent
        )
        self.agents = self.bidding_controller.agents
        self.envs = self.bidding_controller.biddingEnv
        self.pv_generator = self.bidding_controller.pvGenerator

        self.num_agent = len(self.agents)
        self._reset_episode_state()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, episode: Optional[int] = None) -> np.ndarray:
        """
        Resets the environment for a new episode.

        Args:
            episode: Episode index. If None, increments the current episode.

        Returns:
            Initial state vector of shape (STATE_DIM,).
        """
        if episode is not None:
            self.episode = episode
        else:
            self.episode += 1

        self.bidding_controller.reset(episode=self.episode)
        self._reset_episode_state()
        return self._build_state()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advances the environment by one timestep.

        Args:
            action: Scalar alpha (CPA multiplier) output by the PPO policy.

        Returns:
            next_state: State vector after the step.
            reward:     Number of conversions won by the player this tick.
            done:       True if the episode has ended.
            info:       Dict with extra diagnostics (cost, remaining_budget, etc.).
        """
        tick = self.tick_index
        pv_values = self.pv_generator.pv_values[tick]        # (num_pv, num_agent)
        pvalue_sigmas = self.pv_generator.pValueSigmas[tick] # (num_pv, num_agent)

        # --- Build bids for all agents -----------------------------------
        bids = self._collect_bids(pv_values, pvalue_sigmas, action, tick)
        bids = np.array(bids).T   # (num_pv, num_agent)
        bids[bids < 0] = 0

        # --- Overcost adjustment loop (mirrors run_test.py) --------------
        remaining_budgets = np.array([a.remaining_budget for a in self.agents])
        bids, xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, lwc_pit = \
            self._run_auction_with_overcost_guard(pv_values, pvalue_sigmas, bids, remaining_budgets)

        # --- Update budgets ----------------------------------------------
        real_cost = (cost_pit * is_exposed_pit)          # (num_agent, num_pv)
        cost_per_agent = real_cost.sum(axis=1)
        reward_per_agent = conversion_action_pit.sum(axis=1)

        for i, agent in enumerate(self.agents):
            agent.remaining_budget -= cost_per_agent[i]

        # --- Update history arrays ---------------------------------------
        self._update_history(
            pv_values, pvalue_sigmas, bids,
            xi_pit, slot_pit, cost_pit,
            is_exposed_pit, conversion_action_pit, lwc_pit
        )

        # --- Tick bookkeeping -------------------------------------------
        player_cost = cost_per_agent[self.player_index]
        player_reward = float(reward_per_agent[self.player_index])
        player_budget = self.agents[self.player_index].remaining_budget

        self.cumulative_cost += player_cost
        self.cumulative_reward += player_reward
        self.tick_index += 1

        done = (
            self.tick_index >= self.NUM_TICK
            or player_budget < self.envs.min_remaining_budget
        )

        next_state = self._build_state() if not done else np.zeros(self.STATE_DIM)

        info = {
            "tick": tick,
            "cost": player_cost,
            "remaining_budget": player_budget,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_cost": self.cumulative_cost,
            "least_winning_cost": lwc_pit.mean(),
        }

        return next_state, player_reward, done, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_episode_state(self) -> None:
        """Clears per-episode tracking variables."""
        self.tick_index = 0
        self.cumulative_cost = 0.0
        self.cumulative_reward = 0.0

        # History lists — same structure as run_test.py
        self.history_pvalue_infos = []
        self.history_bids = []
        self.history_auction_results = []
        self.history_impression_results = []
        self.history_least_winning_costs = []

        # Running stats for state features (shape: (num_agent,))
        self._sum_bid = np.zeros(self.num_agent)
        self._sum_lwc = np.zeros(self.num_agent)
        self._sum_conv = np.zeros(self.num_agent)
        self._sum_xi = np.zeros(self.num_agent)
        self._sum_pvalue = np.zeros(self.num_agent)
        self._count = np.zeros(self.num_agent)

        self._last3_bids = []
        self._last3_lwc = []
        self._last3_conv = []
        self._last3_xi = []
        self._last3_pvalue = []
        self._tick_volumes = []

    def _collect_bids(
        self,
        pv_values: np.ndarray,
        pvalue_sigmas: np.ndarray,
        player_action: float,
        tick: int,
    ) -> list:
        """
        Builds bids for all agents.
        The player agent uses alpha * pValues; all others use their own strategy.
        """
        bids = []
        for i, agent in enumerate(self.agents):
            if agent.remaining_budget < self.envs.min_remaining_budget:
                bids.append(np.zeros(pv_values.shape[0]))
            elif i == self.player_index:
                # PPO action: alpha multiplier
                bids.append(player_action * pv_values[:, i])
            else:
                bids.append(agent.bidding(
                    tick,
                    pv_values[:, i],
                    pvalue_sigmas[:, i],
                    [x[i] for x in self.history_pvalue_infos],
                    [x[i] for x in self.history_bids],
                    [x[i] for x in self.history_auction_results],
                    [x[i] for x in self.history_impression_results],
                    self.history_least_winning_costs,
                ))
        return bids

    def _run_auction_with_overcost_guard(
        self,
        pv_values: np.ndarray,
        pvalue_sigmas: np.ndarray,
        bids: np.ndarray,
        remaining_budgets: np.ndarray,
    ):
        """
        Mirrors the overcost-adjustment loop from run_test.py.
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
        """Updates all history arrays and running stat accumulators."""
        p = self.player_index
        tick_pv_num = pv_values.shape[0]

        # Running totals for all-time averages (player only)
        tick_bid_mean = bids[:, p].mean()
        tick_lwc_mean = lwc_pit.mean()
        tick_conv_mean = conversion_action_pit[p].mean()
        tick_xi_mean = xi_pit[p].mean()
        tick_pvalue_mean = pv_values[:, p].mean()

        self._sum_bid[p] += tick_bid_mean
        self._sum_lwc[p] += tick_lwc_mean
        self._sum_conv[p] += tick_conv_mean
        self._sum_xi[p] += tick_xi_mean
        self._sum_pvalue[p] += tick_pvalue_mean
        self._count[p] += 1

        # Rolling window (last 3 ticks)
        self._last3_bids.append(tick_bid_mean)
        self._last3_lwc.append(tick_lwc_mean)
        self._last3_conv.append(tick_conv_mean)
        self._last3_xi.append(tick_xi_mean)
        self._last3_pvalue.append(tick_pvalue_mean)
        self._tick_volumes.append(tick_pv_num)

        if len(self._last3_bids) > 3:
            self._last3_bids.pop(0)
            self._last3_lwc.pop(0)
            self._last3_conv.pop(0)
            self._last3_xi.pop(0)
            self._last3_pvalue.pop(0)

        # Standard history arrays (all agents)
        self.history_bids.append(bids.T)
        self.history_least_winning_costs.append(lwc_pit)
        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
        self.history_pvalue_infos.append(pvalue_info)
        auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
        self.history_auction_results.append(auction_info)
        impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
        self.history_impression_results.append(impression_info)

    def _build_state(self) -> np.ndarray:
        """
        Constructs the 16-feature state vector, matching TrainDataGenerator.
        Returns zeros for features that have no history yet (tick 0).
        """
        p = self.player_index
        tick = self.tick_index
        n = self._count[p]

        budget = self.agents[p].budget
        remaining = self.agents[p].remaining_budget
        timeleft = (self.NUM_TICK - tick) / self.NUM_TICK
        bgtleft = remaining / budget if budget > 0 else 0.0

        # All-time averages (shift by 1: use stats from *previous* ticks)
        avg_bid_all = self._sum_bid[p] / n if n > 0 else 0.0
        avg_lwc_all = self._sum_lwc[p] / n if n > 0 else 0.0
        avg_pvalue_all = self._sum_pvalue[p] / n if n > 0 else 0.0
        avg_conv_all = self._sum_conv[p] / n if n > 0 else 0.0
        avg_xi_all = self._sum_xi[p] / n if n > 0 else 0.0

        # Last-3 averages
        avg_bid_last3 = np.mean(self._last3_bids) if self._last3_bids else 0.0
        avg_lwc_last3 = np.mean(self._last3_lwc) if self._last3_lwc else 0.0
        avg_pvalue_last3 = np.mean(self._last3_pvalue) if self._last3_pvalue else 0.0
        avg_conv_last3 = np.mean(self._last3_conv) if self._last3_conv else 0.0
        avg_xi_last3 = np.mean(self._last3_xi) if self._last3_xi else 0.0

        # Current tick traffic features
        if tick < self.NUM_TICK:
            pv_now = self.pv_generator.pv_values[tick][:, p]
            pvalue_mean_now = pv_now.mean()
            tick_volume_now = len(pv_now)
        else:
            pvalue_mean_now = 0.0
            tick_volume_now = 0

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