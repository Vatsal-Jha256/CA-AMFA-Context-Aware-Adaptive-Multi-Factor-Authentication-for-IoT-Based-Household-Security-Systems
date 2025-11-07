from ContextualBandits.ContextualBandit import ContextualBandit
from collections import defaultdict
import datetime
import numpy as np 

class UCBBandit(ContextualBandit):
    """Upper Confidence Bound contextual bandit implementation"""
    
    def __init__(self, factor_names, confidence=2.0):
        super().__init__(factor_names)
        self.confidence = confidence
        self.q_values = {name: defaultdict(float) for name in factor_names}
        self.counts = {name: defaultdict(int) for name in factor_names}
        # Initialize with different values for different factors to encourage exploration
        self.factor_importance = {
            "failed_attempts": 2.0,  # Critical security factor
            "network": 1.8,          # Important security factor
            "user_behavior": 1.5,    # Medium importance
            "time": 1.0,             # Less important
            "motion": 0.8            # Least important
        }
        
    def _context_to_key(self, context):
        """Convert context to a string key for lookup"""
        # Use context's hour and day_of_week instead of current time
        day_of_week = context.get('day_of_week', datetime.datetime.now().weekday())
        hour = context.get('hour', datetime.datetime.now().hour)
        hour_bin = hour // 4
        return f"{day_of_week}_{hour_bin}"
        
    def select_weights(self, context):
        context_key = self._context_to_key(context)
        
        weights = {}
        for name in self.factor_names:
            # If never tried, use factor-specific initial weights
            if self.counts[name][context_key] == 0:
                # Use factor-specific importance or default to 1.5
                weights[name] = self.factor_importance.get(name, 1.5)
                continue
            
            # UCB formula: Q(a) + c * sqrt(ln(total observations for this factor) / N(a))
            exploitation = self.q_values[name][context_key]
            
            # Use factor-specific counts, not total count
            # Calculate total observations for this factor across all contexts
            total_factor_obs = sum(self.counts[name].values()) + 1  # Add 1 to avoid log(0)
            
            # Apply factor-specific importance multiplier to exploration term
            importance_multiplier = self.factor_importance.get(name, 1.0)
            
            exploration = self.confidence * importance_multiplier * np.sqrt(
                np.log(total_factor_obs) / self.counts[name][context_key]
            )
            
            # Bound the weight to reasonable range
            weights[name] = max(0.1, min(3.0, exploitation + exploration))
            
        return weights
        
    def update(self, context, selected_weights, reward):
        context_key = self._context_to_key(context)
        
        # Update values for each factor
        for name in self.factor_names:
            self.counts[name][context_key] += 1
            
            # Incremental average update
            old_q = self.q_values[name][context_key]
            
            # Use a variable learning rate - higher for fewer observations
            # Adjust learning rate based on factor importance
            importance_factor = self.factor_importance.get(name, 1.0)
            learning_rate = importance_factor / self.counts[name][context_key]
            
            # Update using more aggressive learning rate for more responsiveness
            self.q_values[name][context_key] += learning_rate * (reward - old_q)