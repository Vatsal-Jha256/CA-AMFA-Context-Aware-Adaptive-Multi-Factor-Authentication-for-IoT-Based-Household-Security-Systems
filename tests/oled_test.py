#!/usr/bin/env python3
import os
import sys
import time

# Add parent directory to path to import the hardware controller
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware_controller import HardwareController

def test_oled():
    print("===== OLED Display Test (SSD1306) =====")
    print("This test will display various messages on the OLED screen")
    print("Connected to I2C: SDA (Pin 3), SCL (Pin 5)")
    
    try:
        # Initialize hardware controller
        print("Initializing hardware controller...")
        controller = HardwareController()
        # Test series of messages with different formats
        test_messages = [
            "OLED Test",
            "Hello World!",
            "Security System\nInitializing...",
            "Line 1\nLine 2\nLine 3\nLine 4",  # Test multi-line
            "Raspberry Pi\nSecurity System\nTest Complete"
        ]

        for i, message in enumerate(test_messages, 1):
            print(f"\nTest {i}/{len(test_messages)}: Displaying message:")
            print(f"---\n{message}\n---")
            controller.display_message(message)
            time.sleep(3)  # Display each message for 3 seconds
        
        # Clear display at the end
        print("\nClearing display...")
        controller.display_message("")
        
        print("\n? OLED display test completed!")
        
    except Exception as e:
        print(f"\n? OLED test failed with error: {e}")
    finally:
        if 'controller' in locals():
            controller.cleanup()

if __name__ == "__main__":
    test_oled()
