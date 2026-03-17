# Adapting PPO for On-Policy Budget-Constrained Auto-bidding Advertisers
Connie Hong (@connie-hh), Linda Liu (@lindaliu0718)

## 🏛︎ Project Structure (mostly inherited from original AuctionNet codebase)

---

```
├── config                        # Configuration files for setting up the hyperparameters.
├── main_test.py                  # Main entry point for running evaluations.
├── run                           # Core logic for executing tests.

├── simul_bidding_env             # Ad Auction Environment

│   ├── Controller                # Module controlling the simulation flow and logic.
│   ├── Environment               # The auction module.
│   ├── PvGenerator               # The ad opportunity generation module.
│   ├── Tracker                   # Tracking components for monitoring and analysis.
│   │   ├── BiddingTracker.py     # Tracks the bidding process and generates raw data on ad opportunities granularity.
│   │   ├── PlayerAnalysis.py     # Implements metrics to evaluate the performance of user-defined strategies.
│   └── strategy                  # The bidding module (competitors’ strategies).


├── pre_generated_dataset         # Pre-generated dataset.


├── strategy_train_env            # Several baseline bid decision-making algorithms.

│   ├── README_strategy_train.md  # Documentation on how to train the bidding strategy.
│   ├── bidding_train_env         # Core components for training bidding strategies.
│   │   ├── baseline              # Implementation of baseline bid decision-making algorithms.
|   |   |   |── ppo
|   |   |   |   |── ppo.py                  # (!) Our implementation of the PPO agent (train-time)
|   |   |   |   └── two_agent_ppo.py        # (!) Our implementation of the two-agent PPO training process (train-time)

│   │   ├── common                # Common utilities used across modules.
│   │   ├── train_data_generator  # Reads raw data and constructs training datasets.
│   │   ├── offline_eval          # Components required for offline evaluation.
│   │   └── strategy              # Unified bidding strategy interface.
|   |   |   └── ppo_bidding_strategy.py     # (!) Our implementation of the PPO bidding strategy (test-time)
│   ├── data                      # Directory for storing training data.
│   ├── main                      # Main scripts for executing training processes.
│   ├── run                       # Core logic for executing training processes.
|   |   └── run_ppo.py                      # (!) The script to run PPO agent during eval
│   ├── saved_model               # Directory for saving trained models.
│   ├── train_ppo                           # (!) The custom online bidding environment we adapted from the simulation module
│   |   |── ppo_bidding_env.py              
│   |   |── two_agent_ppo_bidding_env.py
```

## 🧑‍💻 Quickstart

---

### Create and activate conda environment
```bash
$ conda create -n AuctionNet python=3.9.12 pip=23.0.1
$ conda activate AuctionNet
```
### Install requirements
```bash
$ pip install -r requirements.txt
```

### Strategy Training

#### Train PPO against 47 other bidders (same bidders in the test environment)
```
python -m strategy_train_env.bidding_train_env.baseline.ppo.ppo
```

### Train PPO Against a Subset of Agents

**Execution Command:**
```bash
python -m strategy_train_env.bidding_train_env.baseline.ppo.ppo_subset
```

---

#### How to Modify the Agent Subset
To change which agents the model trains against, follow these steps:

1.  **Locate the File:** `train_ppo.ppo_bidding_env_subset.py`
2.  **Edit Lines 81–82:** Update the `isinstance` check to include your desired strategy classes.

**Example Configuration:**
```python
# Identify agents by their specific class type
if isinstance(agent, IqlBiddingStrategy) or isinstance(agent, OnlineLpBiddingStrategy):
```

(i.e. change the isintance (agent, Srteargy)) to your liking. 
#### Train PPO against another PPO agent
```
python -m strategy_train_env.bidding_train_env.baseline.ppo.two_agent_ppo
```

### Online Evaluation for single agent

Use the PpoBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
strategy_train_env/bidding_train_env/strategy/__init__.py
from .ppo_bidding_strategy import PpoBiddingStrategy as PlayerBiddingStrategy
```

Set up the hyperparameters for the online evaluation process.
```
config/test.gin
```

Run online evaluation.
```bash
$ python main_test.py
```

### Online Evaluation for dual-agent PPO
```bash
$ python run_test_two_ppo.py
```