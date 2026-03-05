import numpy as np
import matplotlib.pyplot as plt

# Load the cost files
cost1 = np.load('costsboth.npy')
cost2 = np.load('costsonlinelp.npy')
cost3 = np.load('costsiql.npy')

# List of trajectories and their names
data_sets = [
    (cost1, 'PPO (Against IQL & Online LP)'),
    (cost2, 'PPO (Against Online LP)'),
    (cost3, 'PPO (Against IQL)')
]

# Create the plot
plt.figure(figsize=(10, 6))

for data, name in data_sets:
    # Calculate statistics
    mean_val = np.mean(data)
    var_val = np.var(data)
    
    # Create a descriptive label for the legend
    label = f"{name} (μ={mean_val:.2f}, σ²={var_val:.4f})"
    
    plt.plot(data, label=label)

plt.xlabel('Training Step')
plt.ylabel('Cost')
plt.title('Comparison of Cost Trajectories with Statistics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_costs_squareish.png", bbox_inches='tight')