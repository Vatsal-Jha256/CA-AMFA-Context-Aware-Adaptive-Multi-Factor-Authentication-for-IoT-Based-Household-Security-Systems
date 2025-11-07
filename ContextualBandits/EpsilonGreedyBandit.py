import random 
import datetime
from collections import defaultdict
from ContextualBandits.ContextualBandit import ContextualBandit
class EpsilonGreedyBandit(ContextualBandit):
    """Epsilon-greedy contextual bandit implementation"""
    
    def __init__(self, factor_names, epsilon=0.15, learning_rate=0.1):
        super().__init__(factor_names)
        self.epsilon = epsilon  # Increased exploration rate
        self.learning_rate = learning_rate  # Increased learning rate
        self.q_values = {name: defaultdict(float) for name in factor_names}
        self.counts = {name: defaultdict(int) for name in factor_names}
        
    def _context_to_key(self, context):
        """Convert context to a string key for lookup"""
        # Simplified context binning - use context values rather than current time
        day_of_week = context.get('day_of_week', datetime.datetime.now().weekday())
        hour = context.get('hour', datetime.datetime.now().hour)
        hour_bin = hour // 4  # 6 bins of 4 hours each
        return f"{day_of_week}_{hour_bin}"
        
    def select_weights(self, context):
        context_key = self._context_to_key(context)
        
        # With probability epsilon, explore (random weights)
        if random.random() < self.epsilon:
            # More diverse exploration
            return {name: random.uniform(0.2, 2.8) for name in self.factor_names}
        
        # Otherwise, use current best weights
        return {name: self.q_values[name][context_key] 
                if self.counts[name][context_key] > 0 
                else 1.0 
                for name in self.factor_names}
        
    def update(self, context, selected_weights, reward):
        context_key = self._context_to_key(context)
        
        # Update Q-values for each factor
        for name in self.factor_names:
            self.counts[name][context_key] += 1
            
            # Calculate adaptive learning rate - higher for fewer observations
            # This makes the algorithm more responsive to new data
            adaptive_rate = min(1.0, self.learning_rate * (5.0 / self.counts[name][context_key]))
            
            # Update rule: Q = Q + alpha * (R - Q)
            current_q = self.q_values[name][context_key]
            self.q_values[name][context_key] += adaptive_rate * (reward - current_q)
            
            # Ensure weights stay within reasonable bounds
            self.q_values[name][context_key] = max(0.1, min(3.0, self.q_values[name][context_key]))