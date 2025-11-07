# CA-AMFA: Context-Aware Adaptive Multi-Factor Authentication for IoT-Based Household Security Systems

This repository contains the implementation of a Context-Aware Adaptive Multi-Factor Authentication (CA-AMFA) system for IoT-based household security systems. The system dynamically adjusts authentication requirements based on risk assessment using contextual bandit algorithms.

## Quick Start

```bash
# Basic run with reproducible seed
python run_simulation.py --seed 42

# Multiple runs with default seed increment
python run_simulation.py --runs 10 --seed 42

# Use specific seeds for reproducibility
python run_simulation.py --seed_list "42,123,456,789,1010" 

# Set confidence level for statistical intervals (default 0.95)
python run_simulation.py --runs 10 --confidence 0.99

# Generate boxplots to visualize variability
python run_simulation.py --runs 10 --boxplots

# Compare against a different baseline method
python run_simulation.py --runs 10 --compare_with thompson
```

## Overview

The CA-AMFA system leverages a set of risk factors including time-based patterns, user behavior, network conditions, motion detection, and failed login attempts to calculate a risk score. Based on this score, it dynamically selects the appropriate authentication methods (password, OTP, facial recognition) to balance security and usability.

## Hardware Requirements

- Raspberry Pi 3B (or compatible)
- Raspberry Pi Camera Module v1.3
- SSD1306 OLED Display
- Tower Pro MG995 Servo Motor for lock control
- 4×4 Matrix Keypad
- PC for facial recognition processing

## Software Setup

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. Enable I2C and Camera Interface

```bash
sudo raspi-config
# Select Interface Options > I2C > Enable
# Select Interface Options > Camera > Enable
```

### 3. Hardware Connections

Refer to the physical connections section in the hardware_setup.md file for detailed wiring instructions.

## Project Structure

- **ContextualBandits/**: Implementation of various bandit algorithms
- **RiskAssessment/**: Risk factor implementations
- **database/**: Database management
- **models/**: Serialized model files
- **results/**: Performance metrics and evaluation results
- **simulation/**: Simulation framework for testing
- **tests/**: Component test scripts
- **utils/**: Utility functions

## Running the System

### Main Application

```bash
python main.py
```

### Testing Components

```bash
# Test the camera
python tests/test_camera.py

# Test the OLED display
python tests/oled_test.py

# Test the keypad
python tests/keypad_test.py

# Test the servo motor
python tests/servo_test.py
```

## Usage

1. **Registration**: Use the keypad to register a new user (option 5)
2. **Authentication**: Log in with your credentials (option 4)
3. **Risk Assessment**: The system dynamically determines required authentication factors
4. **Authentication Factors**:
   - PIN entry via keypad
   - OTP verification
   - Facial recognition

## License

[MIT License](LICENSE)
