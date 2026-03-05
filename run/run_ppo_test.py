import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from train_ppo_env import PpoBiddingEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # 1. Initialize your environment
    env = PpoBiddingEnv(player_index=0)
    
    # (Optional) Verify that the env matches the gym interface
    # check_env(env)

    # 2. Initialize the PPO Policy
    # n_steps=2048 means it will collect ~42 full episodes (48 ticks each) 
    # before performing a gradient update.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_auction_logs/",
        learning_rate=3e-4,
        gamma=0.99, # High gamma because bidding is a long-term budget game
        n_steps=2048,
        batch_size=64
    )

    # 3. Start Learning
    # This replaces the 'for episode in range(num_episode)' loop from run_test
    logger.info("Starting PPO Training...")
    model.learn(total_timesteps=100000)

    # 4. Save the trained policy
    model.save("ppo_bidding_policy_v1")
    logger.info("Model Saved Successfully.")

if __name__ == "__main__":
    train()