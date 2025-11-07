import time
import random
import psutil
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SecuritySimulation:
    def __init__(self, hardware, mfa, duration_hours=24):
        self.hardware = hardware
        self.mfa = mfa
        self.duration = duration_hours
        
        # Configure test scenarios
        self.test_users = {
            "normal_user": {
                "access_pattern": "regular",  # 9-5 worker
                "success_rate": 0.95,
                "auth_methods": ["password", "otp"],
                "motion_pattern": "normal"  # Regular entry/exit times
            },
            "suspicious_user": {
                "access_pattern": "random",
                "success_rate": 0.6,
                "auth_methods": ["password", "otp", "face"],
                "motion_pattern": "erratic"  # Random motion activity
            }
        }
        
        # Metrics tracking
        self.metrics = {
            "attempts": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "auth_times": [],
            "power_readings": [],
            "cpu_usage": [],
            "memory_usage": [],
            "motion_events": 0
        }
        
    def simulate_motion(self, user_profile, hour):
        """Simulate PIR motion detection"""
        if user_profile["motion_pattern"] == "normal":
            # More likely during business hours
            if 8 <= hour <= 18:
                return random.random() < 0.7
            return random.random() < 0.1
        else:
            # Erratic pattern
            return random.random() < 0.3
            
    def measure_power(self):
        """Simulate power measurement"""
        # Baseline power (e.g., Raspberry Pi idle)
        base_power = 2.5  # Watts
        
        # Add component power usage
        servo_power = 0.5 if self.hardware.servo_active else 0
        camera_power = 1.2 if self.hardware.camera_enabled else 0
        display_power = 0.3
        pir_power = 0.1
        
        # Add some random variation
        noise = random.uniform(-0.2, 0.2)
        
        return base_power + servo_power + camera_power + display_power + pir_power + noise
        
    def simulate_authentication(self, username, profile, hour):
        """Simulate a single authentication attempt"""
        start_time = time.time()
        
        # Record initial metrics
        initial_power = self.measure_power()
        initial_cpu = psutil.cpu_percent()
        initial_mem = psutil.virtual_memory().percent
        
        # Simulate motion before access attempt
        if self.simulate_motion(profile, hour):
            self.metrics["motion_events"] += 1
            self.hardware.mock_motion = True
            time.sleep(0.5)
            self.hardware.mock_motion = False
        
        # Create authentication context
        context = {
            "time": time.time(),
            "hour": hour,
            "device_id": "test_device_001" if random.random() < 0.8 else "unknown_device",
            "day_of_week": datetime.now().weekday()
        }
        
        # Attempt authentication
        success = self.mfa.authenticate_user(username, context)
        auth_time = time.time() - start_time
        
        # Record metrics
        self.metrics["attempts"] += 1
        if success:
            self.metrics["successful_logins"] += 1
        else:
            self.metrics["failed_logins"] += 1
        
        self.metrics["auth_times"].append(auth_time)
        self.metrics["power_readings"].append(self.measure_power() - initial_power)
        self.metrics["cpu_usage"].append(psutil.cpu_percent() - initial_cpu)
        self.metrics["memory_usage"].append(psutil.virtual_memory().percent - initial_mem)
        
        return success, auth_time