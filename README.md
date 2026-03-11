# Adapting PPO for On-Policy Budget-Constrained Auto-bidding Advertisers

## 🏛︎ Project Structure (inherited from original AuctionNet codebase)

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

#### Train PPO against a subset of agents
```
# TODO
```

#### Train PPO against another PPO agent
```
python -m strategy_train_env.bidding_train_env.baseline.ppo.two_agent_ppo
```

### Online Evaluation

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

## 📖 User Case
### Train your own bidding strategy 'awesome_xx'
Refer to the baseline algorithm implementation and complete the following files.
```
├── strategy_train_env
│   ├── bidding_train_env
│   │   ├── baseline
│   │   │   └── awesome_xx
│   │   │       └──awesome_xx.py                # Implement model-related components.
│   │   ├── train_data_generator
│   │   │   └── train_data_generator.py         # Custom-built training Data generation Pipeline.
│   │   └── strategy
│   │       └── awesome_xx_bidding_strategy.py  # Implement Unified bidding strategy interface.
│   ├── main
│   │   └── main_awesome_xx.py                  # Main scripts for executing training processes.
│   └── run
│       └── run_awesome_xx.py                   # Core logic for executing training processes.

```
### Evaluate your own bidding strategy 'awesome_xx'
Use the awesome_xxBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .awesome_xx_bidding_strategy import awesome_xxBiddingStrategy as PlayerBiddingStrategy
```
Run the evaluation process.
```
# Return to the root directory
$ python main_test.py
```


### Generate new dataset
Set the hyperparameters and run the evaluation process.
```
config/test.gin
GENERATE_LOG = True

python main_test.py
```
The newly generated data will be stored in the /data folder.


### Customize new auction environment
We adhere to the programming principles of high cohesion and low coupling to encapsulate each module, making it convenient for users to modify various modules in the auction environment according to their needs.
```
├── simul_bidding_env             # Ad Auction Environment

│   ├── Environment               # The auction module.
│   ├── PvGenerator               # The ad opportunity generation module.
│   ├── Tracker                   
│   │   ├── PlayerAnalysis.py     # Implements metrics to evaluate the performance.
│   └── strategy                  # The bidding module (competitors’ strategies).
```