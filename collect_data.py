"""
Training Data Collection Script
================================
Launches the Udacity simulator in training mode and provides
instructions for collecting driving data for model training.

The simulator records:
- Center, Left, Right camera images (saved to Data/IMG/)
- Driving log CSV (steering, throttle, brake, speed)

Usage:
    python collect_data.py
"""

import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_PATH = os.path.join(SCRIPT_DIR, 'simulator-windows-64', 'Default Windows desktop 64-bit.exe')
DATA_DIR = os.path.join(SCRIPT_DIR, 'Data')
IMG_DIR = os.path.join(DATA_DIR, 'IMG')


def main():
    print('=' * 60)
    print('  TRAINING DATA COLLECTION')
    print('  Autonomous Driving - Capstone Project')
    print('=' * 60)

    # Ensure Data/IMG directory exists
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f'\nData will be saved to: {DATA_DIR}')
    print(f'Images will be saved to: {IMG_DIR}')

    # Check simulator
    if not os.path.exists(SIMULATOR_PATH):
        print(f'\n[ERROR] Simulator not found at:\n  {SIMULATOR_PATH}')
        sys.exit(1)

    # Launch simulator
    print(f'\nLaunching simulator...')
    try:
        process = subprocess.Popen(
            [SIMULATOR_PATH],
            cwd=os.path.dirname(SIMULATOR_PATH)
        )
    except Exception as e:
        print(f'[ERROR] Failed to launch simulator: {e}')
        sys.exit(1)

    print('\n' + '=' * 60)
    print('  INSTRUCTIONS FOR DATA COLLECTION')
    print('=' * 60)
    print("""
    1. In the simulator, select a TRACK (Lake Track recommended)
    2. Click "TRAINING MODE" (NOT Autonomous Mode)
    3. Click the RECORD button (top right)
    4. Choose the save folder: Data/ in your project directory
       Path: {}
    5. DRIVE the car manually:
       - Use arrow keys or WASD to steer
       - Try to stay CENTERED in the lane
       - Drive smoothly (avoid jerky movements)
       - Drive 3-5 laps for good data
    6. Click RECORD again to STOP recording
    7. Close the simulator when done

    TIPS for better training data:
    - Drive at a moderate speed
    - Include recovery driving (steer back from edges)
    - Drive in BOTH directions (clockwise & counter-clockwise)
    - Avoid going off-track

    After collecting data, you'll find:
    - Data/driving_log.csv  (steering angles, speeds, etc.)
    - Data/IMG/             (camera images)
    """.format(DATA_DIR))

    print('Waiting for simulator to close...')
    process.wait()

    # Check if data was collected
    log_path = os.path.join(DATA_DIR, 'driving_log.csv')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
        print(f'\nData collection complete!')
        print(f'  Entries recorded: {len(lines)}')

        img_count = len([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]) if os.path.exists(IMG_DIR) else 0
        print(f'  Images saved: {img_count}')
    else:
        print('\nNo driving_log.csv found. Did you record any data?')
        print(f'Expected location: {log_path}')


if __name__ == '__main__':
    main()
