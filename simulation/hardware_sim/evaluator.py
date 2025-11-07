import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class SecurityEvaluator:
    def __init__(self):
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_comparison(self, simulation, algorithms):
        """Run comparison of different algorithms"""
        results = {}
        
        for algo in algorithms:
            print(f"\nTesting algorithm: {algo}")
            simulation.mfa.active_bandit = algo
            simulation.metrics = self._init_metrics()
            
            # Run simulation
            self._run_simulation(simulation)
            
            # Store results
            results[algo] = self._calculate_metrics(simulation.metrics)
            
        return results
        
    def _init_metrics(self):
        """Initialize clean metrics dictionary"""
        return {
            "attempts": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "auth_times": [],
            "power_readings": [],
            "cpu_usage": [],
            "memory_usage": [],
            "motion_events": 0
        }
        
    def _run_simulation(self, simulation):
        """Run a single simulation"""
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=simulation.duration)
        
        while current_time < end_time:
            for username, profile in simulation.test_users.items():
                hour = current_time.hour
                
                # Determine if access should be attempted
                should_attempt = False
                if profile["access_pattern"] == "regular":
                    should_attempt = 9 <= hour <= 17
                else:
                    should_attempt = random.random() < 0.3
                    
                if should_attempt:
                    simulation.simulate_authentication(username, profile, hour)
                    
            current_time += timedelta(minutes=15)
            time.sleep(0.1)  # Prevent CPU overload
            
    def _calculate_metrics(self, metrics):
        """Calculate final metrics"""
        total_attempts = metrics["attempts"]
        if total_attempts == 0:
            return None
            
        return {
            "accuracy": metrics["successful_logins"] / total_attempts,
            "avg_latency": np.mean(metrics["auth_times"]),
            "avg_power": np.mean(metrics["power_readings"]),
            "peak_power": max(metrics["power_readings"]),
            "avg_cpu": np.mean(metrics["cpu_usage"]),
            "avg_memory": np.mean(metrics["memory_usage"]),
            "motion_events": metrics["motion_events"]
        }
        
    def plot_results(self, results):
        """Generate comparison plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame(results).T
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy and Latency
        sns.barplot(data=df, y=df.index, x="accuracy", ax=axes[0,0])
        axes[0,0].set_title("Authentication Accuracy")
        axes[0,0].set_xlabel("Accuracy")
        
        sns.barplot(data=df, y=df.index, x="avg_latency", ax=axes[0,1])
        axes[0,1].set_title("Average Authentication Time (s)")
        axes[0,1].set_xlabel("Seconds")
        
        # Power Consumption
        sns.barplot(data=df, y=df.index, x="avg_power", ax=axes[1,0])
        axes[1,0].set_title("Average Power Consumption (W)")
        axes[1,0].set_xlabel("Watts")
        
        # Resource Usage
        resource_df = pd.DataFrame({
            'CPU Usage (%)': df['avg_cpu'],
            'Memory Usage (%)': df['avg_memory']
        }, index=df.index)
        resource_df.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title("Resource Usage")
        axes[1,1].set_xlabel("Algorithm")
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comparison_{timestamp}.png")
        plt.close()
        
        # Save detailed metrics
        df.to_csv(f"{self.results_dir}/metrics_{timestamp}.csv")