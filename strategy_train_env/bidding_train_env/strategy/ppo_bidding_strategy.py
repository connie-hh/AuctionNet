import torch
import numpy as np
import os

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class PpoBiddingStrategy(BaseBiddingStrategy):
    """
    PPO Bidding Strategy - loads trained PPO policy for inference
    """

    def __init__(self, budget=100, name="PpoBiddingStrategy", cpa=2, category=1, 
                 model_path=None, device="cpu"):
        super().__init__(budget, name, cpa, category)
        self.category = category
        self.device = torch.device(device)
        
        # Load trained policy
        if model_path is None:
            # Default path - looks for saved model in standard location
            file_name = os.path.dirname(os.path.realpath(__file__))
            dir_name = os.path.dirname(file_name)
            dir_name = os.path.dirname(dir_name)
            model_path = os.path.join(dir_name, "saved_model", "ppo", "ppo_policy_final.pt")
        
        self.policy = self._load_policy(model_path)
        self.policy.eval()  # Set to evaluation mode
        
        # State tracking for feature computation
        self._reset_state_tracking()

    def _load_policy(self, model_path):
        """Load the trained PPO policy network."""
        from strategy_train_env.bidding_train_env.baseline.ppo.ppo import GaussianPolicy
        
        STATE_DIM = 16
        HIDDEN_DIM = 128
        
        policy = GaussianPolicy(STATE_DIM, action_dim=1, hidden_dim=HIDDEN_DIM)
        
        if os.path.exists(model_path):
            policy.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded PPO policy from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using randomly initialized policy.")
        
        return policy

    def reset(self):
        """Reset budget and state tracking at the start of each episode."""
        self.remaining_budget = self.budget
        self._reset_state_tracking()

    def _reset_state_tracking(self):
        """Reset all state tracking variables."""
        self.tick_index = 0
        
        # Running statistics (for all-time averages)
        self._sum_bid = 0.0
        self._sum_lwc = 0.0
        self._sum_conv = 0.0
        self._sum_xi = 0.0
        self._sum_pvalue = 0.0
        self._count = 0
        
        # Rolling window (last 3 ticks)
        self._last3_bids = []
        self._last3_lwc = []
        self._last3_conv = []
        self._last3_xi = []
        self._last3_pvalue = []
        self._tick_volumes = []

    def _build_state(self, timeStepIndex, pValues, historyBid, historyAuctionResult, 
                    historyImpressionResult, historyLeastWinningCost):
        """
        Construct the 16-feature state vector that the PPO policy expects.
        
        State features (matching TrainDataGenerator and PpoBiddingEnv):
            0-1:   timeleft, bgtleft
            2-3:   avg_bid_all, avg_bid_last_3
            4-7:   avg_leastWinningCost_all, avg_pValue_all, avg_conversionAction_all, avg_xi_all
            8-11:  avg_leastWinningCost_last_3, avg_pValue_last_3, avg_conversionAction_last_3, avg_xi_last_3
            12-15: pValue_mean (current), timeStepIndex_volume, last_3_volume, historical_volume
        """
        NUM_TICK = 48
        
        # Time and budget features
        timeleft = (NUM_TICK - timeStepIndex) / NUM_TICK
        bgtleft = self.remaining_budget / self.budget if self.budget > 0 else 0.0
        
        # All-time averages (from previous ticks)
        avg_bid_all = self._sum_bid / self._count if self._count > 0 else 0.0
        avg_lwc_all = self._sum_lwc / self._count if self._count > 0 else 0.0
        avg_pvalue_all = self._sum_pvalue / self._count if self._count > 0 else 0.0
        avg_conv_all = self._sum_conv / self._count if self._count > 0 else 0.0
        avg_xi_all = self._sum_xi / self._count if self._count > 0 else 0.0
        
        # Last-3 averages
        avg_bid_last3 = np.mean(self._last3_bids) if self._last3_bids else 0.0
        avg_lwc_last3 = np.mean(self._last3_lwc) if self._last3_lwc else 0.0
        avg_pvalue_last3 = np.mean(self._last3_pvalue) if self._last3_pvalue else 0.0
        avg_conv_last3 = np.mean(self._last3_conv) if self._last3_conv else 0.0
        avg_xi_last3 = np.mean(self._last3_xi) if self._last3_xi else 0.0
        
        # Current tick features
        pvalue_mean_now = pValues.mean() if len(pValues) > 0 else 0.0
        tick_volume_now = len(pValues)
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

    def _update_state_tracking(self, timeStepIndex, pValues, bids, historyAuctionResult, 
                               historyImpressionResult, historyLeastWinningCost):
        """Update running statistics after bidding."""
        # Compute tick-level statistics
        tick_bid_mean = bids.mean() if len(bids) > 0 else 0.0
        tick_pvalue_mean = pValues.mean() if len(pValues) > 0 else 0.0
        tick_volume = len(pValues)
        
        # Get least winning cost for this tick (if available)
        if len(historyLeastWinningCost) > timeStepIndex:
            tick_lwc_mean = historyLeastWinningCost[timeStepIndex].mean()
        else:
            tick_lwc_mean = 0.0
        
        # Get auction results for this tick (if available)
        if len(historyAuctionResult) > timeStepIndex:
            # historyAuctionResult[tick] shape: (num_pv, 3) with [xi, slot, cost]
            tick_xi_mean = historyAuctionResult[timeStepIndex][:, 0].mean()
        else:
            tick_xi_mean = 0.0
        
        # Get impression results for this tick (if available)
        if len(historyImpressionResult) > timeStepIndex:
            # historyImpressionResult[tick] shape: (num_pv, 2) with [is_exposed, conversion_action]
            tick_conv_mean = historyImpressionResult[timeStepIndex][:, 1].mean()
        else:
            tick_conv_mean = 0.0
        
        # Update running totals
        self._sum_bid += tick_bid_mean
        self._sum_lwc += tick_lwc_mean
        self._sum_pvalue += tick_pvalue_mean
        self._sum_xi += tick_xi_mean
        self._sum_conv += tick_conv_mean
        self._count += 1
        
        # Update rolling windows
        self._last3_bids.append(tick_bid_mean)
        self._last3_lwc.append(tick_lwc_mean)
        self._last3_pvalue.append(tick_pvalue_mean)
        self._last3_xi.append(tick_xi_mean)
        self._last3_conv.append(tick_conv_mean)
        self._tick_volumes.append(tick_volume)
        
        # Keep only last 3
        if len(self._last3_bids) > 3:
            self._last3_bids.pop(0)
            self._last3_lwc.pop(0)
            self._last3_pvalue.pop(0)
            self._last3_xi.pop(0)
            self._last3_conv.pop(0)
        
        self.tick_index += 1

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period using trained PPO policy.

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        # Build state vector
        state = self._build_state(
            timeStepIndex, pValues, historyBid, historyAuctionResult,
            historyImpressionResult, historyLeastWinningCost
        )
        
        # Get action (alpha) from policy
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            mu, std = self.policy(state_tensor)
            # Use mean (deterministic) for evaluation
            alpha = mu.item()
        
        # Cap alpha at 1.5x target CPA (safety mechanism like OnlineLp)
        alpha = min(self.cpa * 1.5, alpha)
        
        # Compute bids
        bids = alpha * pValues
        
        # Update state tracking for next timestep
        self._update_state_tracking(
            timeStepIndex, pValues, bids, historyAuctionResult,
            historyImpressionResult, historyLeastWinningCost
        )
        
        return bids