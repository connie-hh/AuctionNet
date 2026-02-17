import numpy as np
import torch
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.ppo.ppo import PPO

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_PPOModel():
    ppo = PPO("./data/traffic/")
    ppo.train("saved_model/PPOTest")


def run_PPO():
    """
    Run PPO model training and evaluation.
    """
    train_PPOModel()


if __name__ == '__main__':
    run_PPO()