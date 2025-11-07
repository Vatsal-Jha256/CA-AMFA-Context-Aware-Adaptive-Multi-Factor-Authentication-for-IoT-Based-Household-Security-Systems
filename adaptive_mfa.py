import os 
import psutil 
from RiskAssessment.FailedAttemptsRiskFactor import FailedAttemptsRiskFactor
from RiskAssessment.MotionActivityRiskFactor import MotionActivityRiskFactor
from RiskAssessment.NetworkBasedRiskFactor import NetworkBasedRiskFactor
from RiskAssessment.TimeBasedRiskFactor import TimeBasedRiskFactor
from RiskAssessment.UserBehaviorRiskFactor import UserBehaviorRiskFactor
import logging 

from ContextualBandits.EpsilonGreedyBandit import EpsilonGreedyBandit
from ContextualBandits.ThompsonSamplingBandit import ThompsonSamplingBandit
from ContextualBandits.UCBBandit import UCBBandit
import threading    
import time 
import json 
import csv 
import datetime

from database.db import Database
from utils.security import SecurityUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("security_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdaptiveMFA")

class AdaptiveMFA:
    """Main class for adaptive multi-factor authentication"""
    
    def __init__(self, hardware_controller, low_threshold=0.3, high_threshold=0.53, db_path="security_system.db"):
        self.hardware = hardware_controller
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Initialize database
        self.db = Database(db_path)
        
        # Initialize security utilities with database reference
        self.security = SecurityUtils(self.db)
        
        # Migrate users from JSON if database is empty
        if not self.db.get_all_users():
            self.db.migrate_from_json()
    
        
        # Initialize risk factors
        self.risk_factors = {
            "time": TimeBasedRiskFactor(weight=1.0),
            "failed_attempts": FailedAttemptsRiskFactor(weight=1.5),
            "user_behavior": UserBehaviorRiskFactor(weight=1.2),
            "motion": MotionActivityRiskFactor(weight=0.8),
            "network": NetworkBasedRiskFactor(weight=0.5)
        }
        
        # Initialize bandits
        factor_names = list(self.risk_factors.keys())
        self.bandit_algorithms = {
            "epsilon_greedy": self.load_or_create_bandit(EpsilonGreedyBandit, factor_names),
            "thompson": self.load_or_create_bandit(ThompsonSamplingBandit, factor_names),
            "ucb": self.load_or_create_bandit(UCBBandit, factor_names)
        }
        
        # Default bandit
        self.active_bandit = "epsilon_greedy"
        
        # Metrics
        self.metrics = {
            "attempts": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "false_rejections": 0,
            "false_authentications": 0,
            "auth_times": []
        }
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        logger.info("Adaptive MFA system initialized")
        
    def load_or_create_bandit(self, bandit_class, factor_names):
        """Load a bandit from file or create a new one"""
        class_name = bandit_class.__name__
        model_path = f"models/{class_name}.pkl"
        
        # Ensure directory exists
        os.makedirs("models", exist_ok=True)
        
        model = bandit_class.load_model(model_path)
        if model is None:
            model = bandit_class(factor_names)
            model.save_model(model_path)
            
        return model
    
    def calculate_risk_score(self, context):
        """Calculate the current risk score based on all factors and their weights"""
        # Get current weights from active bandit
        weights = self.bandit_algorithms[self.active_bandit].select_weights(context)
        
        # Calculate weighted risk score
        total_weight = 0
        weighted_sum = 0
        
        for name, factor in self.risk_factors.items():
            factor.weight = weights[name]  # Apply bandit-selected weight
            risk_value = factor.get_normalized_value(context)
            weighted_sum += factor.weight * risk_value
            total_weight += factor.weight
            
            # Record risk factor values for analysis
            self.db.record_risk_factor(
                name,
                factor.get_raw_value(context),
                risk_value,
                factor.weight,
                context
            )
            
        # Normalize to 0-1 range
        if total_weight > 0:
            normalized_score = weighted_sum / total_weight
        else:
            normalized_score = 0.5  # Default to medium risk if no weights
            
        return normalized_score, weights
        
    def determine_auth_methods(self, risk_score):
        """Determine which authentication methods to use based on risk score"""
        if risk_score < self.low_threshold:
            return ["password"]
        elif risk_score < self.high_threshold:
            return ["password", "otp"]
        else:
            return ["password", "otp"]  # Face recognition disabled
            
    def authenticate_user(self, username, context=None):
        """Full authentication process with retry mechanisms"""
        start_time = time.time()
        context = context or {}
        context['user_id'] = username
        
        # Check if user exists
        user = self.db.get_user(username)
        if not user:
            self.hardware.display_message("User not found")
            logger.info(f"Authentication attempt for non-existent user: {username}")
            self.metrics["attempts"] += 1
            self.metrics["failed_logins"] += 1
            return False
        
        # Check rate limiting
        if not self.security.check_rate_limit(username):
            self.hardware.display_message("Account locked\nTry later")
            logger.info(f"Rate limit exceeded for user: {username}")
            time.sleep(2)
            return False
            
        # Calculate risk score
        risk_score, weights = self.calculate_risk_score(context)
        logger.info(f"Risk assessment for {username}: {risk_score:.2f}")
        
        # Show risk score on display
        self.hardware.display_message(f"Risk Score: {risk_score:.2f}\nCalculating auth...")
        time.sleep(1.5)  # Give time to see risk score
        
        # Determine required auth methods
        auth_methods = self.determine_auth_methods(risk_score)
        logger.info(f"Required authentication methods: {auth_methods}")
        
        # Show auth requirements on display
        methods_str = ", ".join(auth_methods)
        self.hardware.display_message(f"Login: {username}\nRequired: {methods_str}")
        time.sleep(1)  # Give user time to read
        # Authenticate with required methods
        auth_success = True
        
        # Password authentication
        if "password" in auth_methods:
            self.hardware.display_message("Enter password:")
            password = self.get_password_input()
            if not self.security.verify_password(password, user["password_hash"]):
                auth_success = False
        
        # OTP authentication with retry
        if auth_success and "otp" in auth_methods:
            max_otp_attempts = 5
            otp_verified = False
            
            for attempt in range(1, max_otp_attempts + 1):
                self.hardware.display_message(f"Enter OTP ({attempt}/{max_otp_attempts}):")
                otp_code = self.get_otp_input()
                
                if self.security.verify_totp(user["otp_secret"], otp_code):
                    otp_verified = True
                    break
                elif attempt < max_otp_attempts:
                    self.hardware.display_message("Invalid OTP\nTry again")
                    time.sleep(1)
            
            if not otp_verified:
                auth_success = False
        
        # Face authentication disabled - camera not working
        
        # Get device info for logging
        device_info = {
            "processor": os.cpu_count(),
            "memory": round(psutil.virtual_memory().total / (1024**3), 1)
        }
        # Record authentication attempt
        self.security.record_attempt(
            username=username,
            success=auth_success,
            risk_score=risk_score,
            methods_used=auth_methods,
            device_info=device_info
        )
        
        # Update metrics
        auth_time = time.time() - start_time
        self.metrics["auth_times"].append(auth_time)
        self.metrics["attempts"] += 1
        
        if auth_success:
            self.metrics["successful_logins"] += 1
            self.hardware.display_message(f"Welcome {username}\nAccess granted")
            self.hardware.control_lock(False)  # Unlock
            
            # Update user's last login time and count
            self.db.update_login(username)
            
            # Record successful access for user behavior analysis
            self.risk_factors["user_behavior"].record_access(username)
            
            # Provide positive feedback to bandit
            self.bandit_algorithms[self.active_bandit].update(context, weights, 1.0)
            self.bandit_algorithms[self.active_bandit].save_model(f"models/{self.active_bandit.title()}.pkl")
            
            time.sleep(3)
            self.hardware.control_lock(True)   # Lock again after delay
        else:
            self.metrics["failed_logins"] += 1
            self.hardware.display_message("Access denied")
            
            # Provide negative feedback to bandit
            self.bandit_algorithms[self.active_bandit].update(context, weights, 0.0)
            self.bandit_algorithms[self.active_bandit].save_model(f"models/{self.active_bandit.title()}.pkl")
            
        return auth_success
        
    
    def save_metrics(self):
        """Save current metrics to database"""
        # Calculate derived metrics
        total_attempts = self.metrics["attempts"]
        if total_attempts > 0:
            accuracy = self.metrics["successful_logins"] / total_attempts
            far = self.metrics["false_authentications"] / total_attempts if "false_authentications" in self.metrics else 0
            frr = self.metrics["false_rejections"] / total_attempts if "false_rejections" in self.metrics else 0
        else:
            accuracy = far = frr = 0
            
        avg_latency = sum(self.metrics["auth_times"]) / len(self.metrics["auth_times"]) if self.metrics["auth_times"] else 0
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Create metrics record
        metrics_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "bandit_algorithm": self.active_bandit,
            "attempts": total_attempts,
            "successful_logins": self.metrics["successful_logins"],
            "failed_logins": self.metrics["failed_logins"],
            "accuracy": accuracy,
            "false_authentication_rate": far,
            "false_rejection_rate": frr,
            "avg_latency": avg_latency,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent
        }
        
        # Save to database
        self.db.save_metrics(metrics_record)
        
        # Also save to CSV for backwards compatibility
        metrics_file = "results/metrics.csv"
        file_exists = os.path.isfile(metrics_file)
        
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_record)

    def get_password_input(self):
        """Get password input from keypad"""
        password = ""
        self.hardware.display_message("Enter password:\n" + "*" * len(password))
        
        while True:
            key = self.hardware.read_keypad()
            if key:
                if key == '#':  # Submit
                    return password
                elif key == '*':  # Backspace
                    if password:
                        password = password[:-1]
                else:
                    password += key
                self.hardware.display_message("Enter password:\n" + "*" * len(password))
            time.sleep(0.1)

    def verify_password(self, username, password):
        """Verify password with rate limiting"""
        # Check rate limit based on database records
        failed_attempts = self.db.get_failed_attempts_count(username, seconds=300)
        if failed_attempts >= 3:  # Max 3 attempts in 5 minutes
            self.hardware.display_message("Account locked\nTry later")
            time.sleep(2)
            return False
            
        # Get user from database
        user = self.db.get_user(username)
        if not user:
            return False
            
        # Verify password
        return self.security.verify_password(password, user["password_hash"])
        
    def get_otp_input(self):
        """Get OTP code from keypad or keyboard"""
        otp = ""
        self.hardware.display_message("Enter OTP:\n" + otp)
        
        # Set up keyboard input
        print("Enter OTP (type on keyboard or use keypad):")
        
        timeout = 60  # 60 seconds timeout for OTP entry
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check hardware keypad
            key = self.hardware.read_keypad()
            if key:
                if key == '#':  # Submit on keypad
                    print(f"OTP entered: {otp}")
                    return otp
                elif key == '*':  # Backspace on keypad
                    if otp:
                        otp = otp[:-1]
                        print(f"OTP (backspace): {otp}")
                elif key.isdigit() and len(otp) < 6:  # Standard 6-digit OTP
                    otp += key
                    print(f"OTP (keypad): {otp}")
                
                self.hardware.display_message("Enter OTP:\n" + otp)
            
            # Check for keyboard input (non-blocking)
            if os.name == 'nt':  # Windows
                import msvcrt
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8', errors='ignore')
                    self._process_keyboard_input(char, otp)
            else:  # Unix/Linux
                import select
                import sys
                if select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    self._process_keyboard_input(char, otp)
            
            time.sleep(0.1)  # Short sleep to reduce CPU usage
        
        # If timeout occurs
        self.hardware.display_message("OTP timeout")
        print("OTP entry timed out")
        return otp

    def _process_keyboard_input(self, char, otp):
        """Process a single character from keyboard input for OTP"""
        if char == '\r' or char == '\n':  # Enter key
            print(f"OTP submitted: {otp}")
            return otp
        elif char == '\b' or char == '\x7f':  # Backspace
            if otp:
                otp = otp[:-1]
                print(f"OTP (backspace): {otp}")
        elif char.isdigit() and len(otp) < 6:
            otp += char
            print(f"OTP (keyboard): {otp}")
        
        self.hardware.display_message("Enter OTP:\n" + otp)
        return None

    def verify_otp(self, username, otp):
        """Verify OTP using TOTP"""
        user = self.db.get_user(username)
        if not user:
            return False
            
        return self.security.verify_totp(user["otp_secret"], otp)
        
    def verify_face(self, username):
        """Verify face - DISABLED: Camera not working"""
        # Face recognition disabled - always return True to allow authentication
        logger.info("Face verification disabled - camera not available")
        return True
    
    def enroll_user(self, username, password, capture_face=True):
        """Enroll a new user with improved face capture reliability"""
        # Check if user already exists
        if self.db.get_user(username):
            self.hardware.display_message("User exists")
            time.sleep(1)
            return False
        
        # Hash the password
        password_hash = self.security.hash_password(password)
        
        # Generate OTP secret
        otp_secret = self.security.generate_totp_secret()
        
        # Face recognition disabled - camera not working
        face_encoding = None
        
        # Create user in database
        success = self.db.create_user(
            username,
            password_hash,
            otp_secret,
            face_encoding
        )
        
        if success:
            self.hardware.display_message(f"User {username}\nenrolled")
            
            # Display QR code for OTP setup with improved visibility
            totp_uri = self.security.get_totp_qr(username, otp_secret)
            self.hardware.display_message("TOTP Secret:\n" + otp_secret)
            time.sleep(3)  # Give time to read the secret
            
            # Display actual QR code if hardware supports it
            self.hardware.display_qr_code(totp_uri)
            time.sleep(5)
        else:
            self.hardware.display_message("Enrollment\nfailed")
            time.sleep(2)
            
        return success
    
    def delete_user(self, username, admin_password):
        """Delete a user (requires admin verification)"""
        # Verify admin credentials
        admin = self.db.get_user("admin")
        if not admin or not self.security.verify_password(admin_password, admin["password_hash"]):
            self.hardware.display_message("Admin auth\nfailed")
            time.sleep(2)
            return False
        
        # Delete user
        success = self.db.delete_user(username)
        if success:
            self.hardware.display_message(f"User {username}\ndeleted")
        else:
            self.hardware.display_message("User not found")
        time.sleep(2)
        return success
    
    def monitor_loop(self):
        """Background monitoring thread"""
        last_save = time.time()
        save_interval = 300  # Save metrics every 5 minutes
        
        while self.running:
            # Check for motion
            if self.hardware.read_motion():
                self.risk_factors["motion"].record_motion()
                
            # Save metrics periodically
            if time.time() - last_save > save_interval:
                self.save_metrics()
                last_save = time.time()
                
            time.sleep(1)

    def shutdown(self):
        """Clean shutdown of the system"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.save_metrics()
        
        # Restore terminal settings
        self.restore_keyboard_input()
        
        logger.info("AdaptiveMFA system shutdown")


    def setup_keyboard_input(self):
        """Set up system for non-blocking keyboard input"""
        if os.name == 'nt':  # Windows
            # Windows doesn't need special setup for msvcrt
            pass
        else:  # Unix/Linux
            # For Unix-like systems, set terminal to raw mode
            import sys
            import termios
            import tty
            
            # Save terminal settings
            self.old_settings = termios.tcgetattr(sys.stdin)
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
        
        print("Keyboard input setup complete.")

    def restore_keyboard_input(self):
        """Restore terminal settings"""
        if os.name != 'nt':  # Only needed for Unix/Linux
            import sys
            import termios
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
        print("Terminal settings restored.")
