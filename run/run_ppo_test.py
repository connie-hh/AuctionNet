#Modify run_test.py to train the PPO agent and evaluate its performance.
from run.run_test import run_test
from train_ppo import train_ppo_agent

def run_ppo_test():
    # Train the PPO agent
    train_ppo_agent()

    # Evaluate the PPO agent
    run_test()
    