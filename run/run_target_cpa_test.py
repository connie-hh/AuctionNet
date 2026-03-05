import sys
import time
import math
import os
from typing import List, Optional

import gin
import numpy as np
import psutil

from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker
from simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis
from simul_bidding_env.Controller.Controller import Controller
# from collections import Iterable
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def initialize_player_agent() -> Optional['PlayerAgent']:
    """Initializes the PlayerAgent with appropriate strategy.

    Returns:
        Optional[PlayerAgent]: An instance of PlayerAgent.
    """
    try:
        # raise ImportError("")
        from bidding_train_env.strategy import PlayerBiddingStrategy as PlayerAgent
        logger.info("Successfully imported PlayerBiddingStrategy as PlayerAgent.")
        return PlayerAgent()
    except ImportError as import_error:
        logger.error(f"Failed to import PlayerAgent: {import_error}")
        try:
            from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy as PlayerAgent
            import numpy as np
            agent = PlayerAgent(exp_tempral_ratio=np.ones(48))
            agent.name += "0"
            logger.info("Successfully loaded PidBiddingStrategy as PlayerAgent in local run mode.")
            return agent
        except ImportError as e:
            logger.error(f"Failed to import PidBiddingStrategy: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during PlayerAgent initialization: {e}")
        sys.exit(1)




def get_winner(slot_pit: np.ndarray) -> np.ndarray:
    """Determines the winners for each slot in the pit.

    """
    slot_pit = slot_pit.T
    num_pv, num_agent = slot_pit.shape
    num_slot = 3
    winner = np.full((num_pv, num_slot), -1, dtype=int)

    for pos in range(1, num_slot + 1):
        winning_agents_indices = np.argwhere(slot_pit == pos)
        if winning_agents_indices.size > 0:
            pv_indices, agent_indices = winning_agents_indices.T
            winner[pv_indices, pos - 1] = agent_indices
    return winner


def adjust_over_cost(bids: np.ndarray, over_cost_ratio: np.ndarray, envs_slots: List[int],winner_pit) -> None:
    """Adjusts the bids to prevent overcost.

    """
    overcost_agent_indices = np.where(over_cost_ratio > 0)[0]
    for agent_index in overcost_agent_indices:
        for i, coefficient in enumerate(envs_slots):
            winner_indices = winner_pit[:, i]
            pv_indices = np.where(winner_indices == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_to_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_to_drop > 0:
                dropped_pv_indices = rng.choice(pv_indices, num_to_drop, replace=False)
                bids[dropped_pv_indices, agent_index] = 0


def initialize_player_analysis() -> PlayerAnalysis:
    """Initializes the PlayerAnalysis instance.

    """
    return PlayerAnalysis("player_analysis")


def log_memory_usage() -> None:
    """Logs the current memory usage of the process."""
    pid = os.getpid()
    current_process = psutil.Process(pid)
    memory_info = current_process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"Memory usage: {memory_mb:.2f} MB")


@gin.configurable
def run_test(generate_log: bool = False,
             num_episode: int = 2,
             num_tick: int = 48,
             player_index: int = 0,
             ):
    """Runs the bidding simulation test.

    Args:
        generate_log (bool): Flag to generate logs.
        num_episode (int): Number of episodes to run.
        num_tick (int): Number of ticks per episode.
        player_index (int): Index of the player agent.

    Returns:
        ReturnType: Result of the player analysis.
    """

    player_agent = initialize_player_agent()
    player_analysis = initialize_player_analysis()
    bidding_controller = Controller(player_index=player_index, player_agent=player_agent)

    agents = bidding_controller.agents
    num_agent = len(agents)
    agents_category = np.array([agent.category for agent in agents])
    agents_cpa = np.array([agent.cpa for agent in agents])

    envs = bidding_controller.biddingEnv
    pv_generator = bidding_controller.pvGenerator
    train_data_tracker = BiddingTracker("train_data_generate_tracker")
    logger.info(
        f"playerAgentName:{agents[player_index].name} playerAgentCpa:{agents[player_index].cpa} playerAgentBudget:{agents[player_index].budget} playerAgentCategory:{agents[player_index].category} PlayerIndex:{player_index}")

    total_pv_num = 0
    begin_time = time.time()

    for episode in range(num_episode):
        logger.info(f" PlayerIndex:{player_index} episode:{episode} evaluate")
        if generate_log:
            train_data_tracker.reset()

        rewards = np.zeros(num_agent)
        costs = np.zeros(num_agent)
        budgets = np.array([agent.budget for agent in agents])

        history_pvalue_infos = []
        history_bids = []
        history_auction_results = []
        history_impression_results = []
        history_least_winning_costs = []

        bidding_controller.reset(episode=episode)


        for tick_index in range(num_tick):
            pv_values = pv_generator.pv_values[tick_index]
            pvalue_sigmas = pv_generator.pValueSigmas[tick_index]

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

            ratio_max = None
            while ratio_max is None or ratio_max > 0:
                if ratio_max and ratio_max > 0:
                    over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                    adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients,winner_pit)

                xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, least_winning_cost_pit, market_price_pit = \
                    envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

                real_cost = cost_pit * is_exposed_pit
                cost = real_cost.sum(axis=1)
                reward = conversion_action_pit.sum(axis=1)

                winner_pit = get_winner(slot_pit)
                over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                ratio_max = over_cost_ratio.max()

            for i, agent in enumerate(agents):
                agent.remaining_budget -= cost[i]

            rewards += reward
            costs += cost

            history_bids.append(bids.transpose())
            history_least_winning_costs.append(least_winning_cost_pit)
            pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
            history_pvalue_infos.append(pvalue_info)
            auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
            history_auction_results.append(auction_info)
            impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
            history_impression_results.append(impression_info)

            if generate_log:
                train_data_tracker.train_logging(
                    episode, tick_index, pv_values, budgets, agents_cpa, agents_category,
                    remaining_budget_list, total_pv_num, pvalue_sigmas, bids,
                    xi_pit, slot_pit, cost_pit, is_exposed_pit,
                    conversion_action_pit, least_winning_cost_pit, done_list
                )

            if player_analysis:
                tick_win_pv = np.sum(is_exposed_pit[player_index])
                tick_compete_pv = len(xi_pit[player_index])
                tick_all_win_bid = np.sum(bids[:, player_index] * is_exposed_pit[player_index])
                bid_mean = np.mean(bids[:, player_index])
                player_analysis.logging_player_tick(
                    episode, tick_index, player_index, agents_cpa[player_index],
                    budgets[player_index], reward[player_index],
                    cost[player_index], tick_compete_pv, tick_win_pv,
                    tick_all_win_bid, bid_mean
                )

        if generate_log:
            train_data_tracker.generate_train_data(f"data/log/{episode}.csv")


    end_time = time.time()
    logger.info(f"Total time elapsed: {end_time - begin_time} seconds")
    # log_memory_usage()

    player_analysis.player_multi_episode(agents[player_index].name)
    result = player_analysis.get_return_res(
        agents[player_index].name,
        player_index,
        agents[player_index].category
    )
    return result




if __name__ == "__main__":
    pass
