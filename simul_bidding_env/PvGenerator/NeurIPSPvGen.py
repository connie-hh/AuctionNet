import numpy as np
import random
import logging
from typing import Tuple, List

# Constants
SEED = 1019

# Configure seed and print options
random.seed(SEED)
np.random.seed(SEED)
np.set_printoptions(suppress=True, precision=4)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeurIPSPvGen:
    """Generates PV values and sigmas for agents over ticks."""

    def __init__(self, episode: int = 0, num_tick=48, num_agent=48,num_agent_category = 8, num_category=6,pv_num=500000):
        self.NUM_TICK = num_tick
        self.NUM_AGENT = num_agent
        self.NUM_AGENT_CATEGORY = num_agent_category
        self.NUM_CATEGORY = num_category
        self.PV_NUM = pv_num
        self.episode = episode
        self.traffic_num_ratio_base = self.load_traffic_num_ratio_base()
        self.pvalue_mean_ratio_base = self.load_pvalue_mean_ratio_base()

        self.traffic_num_ratio_scale = 0.4
        self.traffic_num_ratio_window = 4

        self.pvalue_mean_base = 0.0005
        self.pvalue_mean_by_diff_category_ratio_scale = 0.7
        self.pvalue_mean_by_diff_category_ratio_window = 8
        self.pvalue_mean_by_same_category_ratio_scale = 0.5
        self.pvalue_mean_by_same_category_ratio_window = 8

        self.pvalue_std_ratio_mean = 0.5
        self.pvalue_std_ratio_std = 0.1
        self.pvalue_std_ratio_by_diff_tick_scale = 0.2

        self.pvalue_sigma_mean = 0.15
        self.pvalue_sigma_std = 0.06

        self.pv_values, self.pValueSigmas = self.generate()

    def generate(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generates PV values and sigmas."""

        traffic_num = self.generate_traffic_num()
        pvalue_mean_all_agent = self.generate_pvalue_mean()
        pvalue_std_all_agent = self.generate_pvalue_std_ratio() * pvalue_mean_all_agent
        pvalues = self.generate_pvalues(pvalue_mean_all_agent, pvalue_std_all_agent, traffic_num)
        pvalue_sigmas = self.generate_pvalue_sigma(pvalues)

        return pvalues, pvalue_sigmas

    def generate_traffic_num(self) -> np.ndarray:
        """Generates traffic number distribution."""
        traffic_num_ratio = self.generate_perturb_with_normalize(
            self.traffic_num_ratio_base,
            self.traffic_num_ratio_scale,
            self.traffic_num_ratio_scale,
            self.traffic_num_ratio_window,
            self.episode
        )
        return (traffic_num_ratio * self.PV_NUM).astype(np.int64)

    def generate_pvalue_mean(self) -> np.ndarray:
        """Generates pvalue mean distribution for all agents."""
        pvalue_mean_by_diff_category_ratio = self.calculate_pvalue_mean_by_category()
        pvalue_mean_all_agent_ratio = self.calculate_pvalue_mean_all_agents(pvalue_mean_by_diff_category_ratio)
        pvalue_mean_all_agent = pvalue_mean_all_agent_ratio * self.pvalue_mean_base

        pvalue_mean_all_agent[pvalue_mean_all_agent < 0] = self.pvalue_mean_base * 0.01
        pvalue_mean_all_agent[pvalue_mean_all_agent > 1] = 1

        return pvalue_mean_all_agent

    def calculate_pvalue_mean_by_category(self) -> np.ndarray:
        """Calculates pvalue mean by category."""
        pvalue_mean_by_diff_category_ratio = np.empty((self.NUM_CATEGORY, self.NUM_TICK))
        for i in range(self.NUM_CATEGORY):
            pvalue_mean_by_diff_category_ratio[i] = self.generate_perturb_no_normalize(
                self.pvalue_mean_ratio_base,
                self.pvalue_mean_by_diff_category_ratio_scale,
                self.pvalue_mean_by_diff_category_ratio_scale,
                self.pvalue_mean_by_diff_category_ratio_window,
                self.episode * 11111 + i
            )
        return pvalue_mean_by_diff_category_ratio

    def calculate_pvalue_mean_all_agents(self, pvalue_mean_by_diff_category_ratio: np.ndarray) -> np.ndarray:
        """Calculates pvalue mean for all agents."""
        pvalue_mean_all_agent_ratio = np.empty((self.NUM_AGENT, self.NUM_TICK))
        for i in range(self.NUM_AGENT):
            cate = i // self.NUM_AGENT_CATEGORY
            cate_pvalue_mean = pvalue_mean_by_diff_category_ratio[cate]

            if i % self.NUM_AGENT_CATEGORY == 0:
                pvalue_mean_all_agent_ratio[i] = cate_pvalue_mean
            else:
                pvalue_mean_all_agent_ratio[i] = self.generate_perturb_no_normalize(
                    cate_pvalue_mean,
                    self.pvalue_mean_by_same_category_ratio_scale,
                    self.pvalue_mean_by_same_category_ratio_scale,
                    self.pvalue_mean_by_same_category_ratio_window,
                    self.episode * 11111 + i
                )
        return pvalue_mean_all_agent_ratio

    def generate_pvalue_std_ratio(self) -> np.ndarray:
        """Generates pvalue std ratio for all agents."""
        rg = np.random.default_rng(seed=self.episode)
        pvalue_std_agent_mean = rg.normal(self.pvalue_std_ratio_mean, self.pvalue_std_ratio_std, self.NUM_AGENT)
        pvalue_std_agent_mean[pvalue_std_agent_mean < 0] = 0.1
        pvalue_std_agent_mean[pvalue_std_agent_mean > 1] = 1
        all_values = rg.normal(
            pvalue_std_agent_mean[:, np.newaxis],
            self.pvalue_std_ratio_by_diff_tick_scale * pvalue_std_agent_mean[:, np.newaxis],
            (self.NUM_AGENT, self.NUM_TICK)
        )
        all_values[all_values < 0] = 0.1
        all_values[all_values > 1] = 1
        return all_values

    def generate_pvalues(self, pvalue_mean_all_agent: np.ndarray, pvalue_std_all_agent: np.ndarray,
                         traffic_num: np.ndarray) -> List[np.ndarray]:
        """Generates pvalues for all ticks."""
        rg = np.random.default_rng(seed=self.episode)
        traffic_list = []

        for i in range(self.NUM_TICK):
            num_traffic = traffic_num[i]
            means = pvalue_mean_all_agent[:, i]
            stds = pvalue_std_all_agent[:, i]
            tick_traffic = rg.normal(means, stds, (int(num_traffic), self.NUM_AGENT))
            tick_traffic[tick_traffic < 0] = 0
            tick_traffic[tick_traffic > 1] = 1
            traffic_list.append(tick_traffic)

        return traffic_list

    def generate_pvalue_sigma(self, pvalues: List[np.ndarray]) -> List[np.ndarray]:
        """Generates pvalue sigmas."""
        pvalue_sigmas = [np.empty_like(array) for array in pvalues]
        rg = np.random.default_rng(seed=self.episode)
        sigma_ratio_mean_agent = rg.normal(self.pvalue_sigma_mean, self.pvalue_sigma_std, self.NUM_AGENT)
        sigma_ratio_mean_agent[sigma_ratio_mean_agent < 0] *= -1
        sigma_ratio_mean_agent[sigma_ratio_mean_agent > 0.3] = 0.2
        means = sigma_ratio_mean_agent[np.newaxis, :]
        stds = (sigma_ratio_mean_agent / 2)[np.newaxis, :]

        for i, array in enumerate(pvalues):
            shape = array.shape
            rg = np.random.default_rng(seed=self.episode * 120000 + i)
            sampled_array = rg.normal(means, stds, shape)
            sampled_array[sampled_array < 0] *= -1
            sampled_array[sampled_array > 0.3] = 0.3
            pvalue_sigmas[i] = sampled_array * array

        return pvalue_sigmas

    def reset(self, episode: int = 0):
        """Resets the generator with a new episode."""
        self.__init__(
            episode=episode,
            num_tick=self.NUM_TICK,
            num_agent=self.NUM_AGENT,
            num_agent_category=self.NUM_AGENT_CATEGORY,
            num_category=self.NUM_CATEGORY,
            pv_num=self.PV_NUM,
        )

    def load_traffic_num_ratio_base(self) -> np.ndarray:
        """Loads the base traffic number ratio."""
        return np.array([0.02301227869859104, 0.010762491580186426, 0.008821901210171106, 0.004815771833139463,
                         0.004798848017825916, 0.0033286522466030154, 0.003393029871737658, 0.003033162880076294,
                         0.004298719169969498, 0.004862999633901365, 0.011135644383826103, 0.012360252404100745,
                         0.018506556921931334, 0.017290990682316758, 0.02611926027547618, 0.02405014108139558,
                         0.036408735251439284, 0.03309464578051337, 0.03811129136190582, 0.032375488041034066,
                         0.03719169795965519, 0.03287603678685833, 0.033527553577337985, 0.025498437500844198,
                         0.023719894875409323, 0.02221396205784743, 0.026215595281236393, 0.020075013906798774,
                         0.02223287715563599, 0.020495297399888097, 0.022686423942899624, 0.020808760368362482,
                         0.02412421281824744, 0.020575376772894775, 0.019990263429615227, 0.01648567735506885,
                         0.01745555889928084, 0.016890210294343122, 0.020425629711104402, 0.0197873391643239,
                         0.024950761402728776, 0.025119778635971764, 0.03110131195417271, 0.028594527343256595,
                         0.030060816887030082, 0.02565065380963528, 0.02538045801878989, 0.025285011364621513])

    def load_pvalue_mean_ratio_base(self) -> np.ndarray:
        """Loads the base pvalue mean ratio."""
        return np.array([1] * self.NUM_TICK)

    def generate_perturb_with_normalize(self, num_list: np.ndarray, upscale: float, downscale: float, window_size: int,
                                        episode: int) -> np.ndarray:
        """Generates perturb with normalization."""
        rg = np.random.default_rng(seed=episode)
        original_total = np.sum(num_list)
        random_factors = rg.uniform(1 - downscale, 1 + upscale, len(num_list) // window_size)
        random_factors = np.repeat(random_factors, window_size)
        final_factors = random_factors * (original_total / np.sum(random_factors * num_list))
        return final_factors * num_list

    def generate_perturb_no_normalize(self, num_list: np.ndarray, upscale: float, downscale: float, window_size: int,
                                      episode: int) -> np.ndarray:
        """Generates perturb without normalization."""
        rg = np.random.default_rng(seed=episode)
        random_factors = rg.uniform(1 - downscale, 1 + upscale, len(num_list) // window_size)
        random_factors = np.repeat(random_factors, window_size)
        return random_factors * num_list


def test():
    """Tests the NeurIPSPvGen class."""
    for i in range(7):
        NeurIPS_gen = NeurIPSPvGen(episode=i)
        pv_values = NeurIPS_gen.pv_values
        pv_sigma = NeurIPS_gen.pvalue_sigmas
        logger.info(f"Episode {i}")
        logger.info(f"PV Values: {pv_values[0][0][:10]}")
        logger.info(f"PV Sigmas: {pv_sigma[0][0][:10]}")
        logger.info("----------------")


if __name__ == '__main__':
    test()
