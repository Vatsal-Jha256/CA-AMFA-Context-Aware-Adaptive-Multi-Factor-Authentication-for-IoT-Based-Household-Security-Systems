import RPi.GPIO as GPIO
import time
import sys

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Use the pins from your setup guide
ROWS = [23, 24, 25, 8]
COLS = [7, 12, 16, 20]

KEYS = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D']
]

# Set up row pins as outputs
for row in ROWS:
    GPIO.setup(row, GPIO.OUT)
    GPIO.output(row, GPIO.HIGH)

# Set up column pins as inputs with pull-up
for col in COLS:
    GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Check for direct input mode
direct_input_mode = '--direct' in sys.argv or '-d' in sys.argv

if direct_input_mode:
    print("Direct Input Mode - Enter keys manually")
    print("Enter a key (1-9, 0, A-D, *, #) or 'exit' to quit:")
    try:
        while True:
            user_input = input("Enter key: ").strip().upper()
            if user_input == 'EXIT':
                break
            if user_input in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', '*', '#']:
                print(f"Key entered: {user_input}")
            else:
                print(f"Invalid key: {user_input}. Valid keys: 1-9, 0, A-D, *, #")
    except KeyboardInterrupt:
        print("\nExiting test")
    finally:
        GPIO.cleanup()
else:
    print("Keypad Test Running. Press keys on your keypad.")
    print("Press Ctrl+C to exit.")
    print("Note: If hardware keypad not working, run with --direct or -d flag for manual input")

    try:
        while True:
            key_found = False
            # Ensure all rows start HIGH
            for row in ROWS:
                GPIO.output(row, GPIO.HIGH)
            
            time.sleep(0.001)  # Small delay to stabilize
            
            # Scan rows and columns
            for row_idx, row in enumerate(ROWS):
                GPIO.output(row, GPIO.LOW)  # Set current row to LOW
                time.sleep(0.001)  # Small delay for signal to stabilize
                
                for col_idx, col in enumerate(COLS):
                    if GPIO.input(col) == GPIO.LOW:  # Key is pressed
                        key = KEYS[row_idx][col_idx]
                        print(f"Key pressed: {key} at position ({row_idx},{col_idx})")
                        key_found = True
                        
                        # Wait for key release with debounce
                        debounce_count = 0
                        while GPIO.input(col) == GPIO.LOW:
                            time.sleep(0.01)
                            debounce_count += 1
                            if debounce_count > 50:  # Timeout after 0.5s
                                break
                        
                        # Reset all rows before continuing
                        for r in ROWS:
                            GPIO.output(r, GPIO.HIGH)
                        break
                
                # Reset current row to HIGH before moving to next row
                GPIO.output(row, GPIO.HIGH)
                time.sleep(0.001)  # Small delay between rows
                
                if key_found:
                    break
            
            if not key_found:
                time.sleep(0.05)  # Short delay between scans when no key pressed

    except KeyboardInterrupt:
        print("\nExiting test")
    finally:
        GPIO.cleanup()
