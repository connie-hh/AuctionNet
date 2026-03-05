import numpy as np
import matplotlib.pyplot as plt

# Load the reward files
rewards1 = np.load('reward_against_iql_onlinelp.npy')
rewards2 = np.load('reward_against_onlinelp.npy')
rewards3 = np.load('rewards_againstiql.npy')

# List of trajectories and their names
data_sets = [
    (rewards1, 'PPO (Against IQL & Online LP)'),
    (rewards2, 'PPO (Against Online LP)'),
    (rewards3, 'PPO (Against IQL)')
]

# Create the plot
plt.figure(figsize=(8, 6))

for data, name in data_sets:
    # Calculate statistics
    mean_val = np.mean(data)
    var_val = np.var(data)
    
    # Create a descriptive label for the legend
    label = f"{name} (μ={mean_val:.2f}, σ²={var_val:.4f})"
    
    plt.plot(data, label=label)

plt.xlabel('Training Step')
plt.ylabel('Reward')
plt.title('Comparison of Reward Trajectories with Statistics')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("all_rewards_squareish.png",bbox_inches='tight')