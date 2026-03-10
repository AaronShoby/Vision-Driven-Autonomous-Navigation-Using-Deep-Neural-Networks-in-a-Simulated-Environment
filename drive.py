"""
Autonomous Driving - Self-Driving Car Controller
================================================
This script loads a trained CNN model and connects to the
TEST DRIVE SIMULATOR. It auto-launches the simulator, receives
camera images, predicts steering angles, and sends control commands.

Usage:
    python drive.py              # Auto-launch simulator + drive
    python drive.py --no-launch  # Skip auto-launch (if simulator is already open)
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import time
import subprocess
import argparse
import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_PATH = os.path.join(SCRIPT_DIR, 'beta_simulator_windows', 'beta_simulator.exe')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'model.h5')
PORT = 4567
SPEED_LIMIT = 25

# ============================================================
# SocketIO Server Setup
# ============================================================
sio = socketio.Server()
app = Flask(__name__)


def img_preprocess(img):
    """
    Preprocess camera image for the CNN model.
    Based on NVIDIA's End-to-End Learning architecture:
    1. Crop - Remove sky (top) and hood (bottom) 
    2. Convert to YUV colorspace (better for lane detection)
    3. Gaussian blur to reduce noise
    4. Resize to model input size (200x66)
    5. Normalize pixel values to [0, 1]
    """
    img = img[60:135, :, :]                        # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)     # Color space
    img = cv2.GaussianBlur(img, (3, 3), 0)         # Blur
    img = cv2.resize(img, (200, 66))               # Resize
    img = img / 255                                # Normalize
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    """Handle telemetry data from the simulator."""
    if data is None:
        return
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image, verbose=0)[0][0])
    throttle = 1.0 - speed / SPEED_LIMIT
    print(f'Steering: {steering_angle:+.4f} | Throttle: {throttle:.4f} | Speed: {speed:.2f}')
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    """Handle new connection from the simulator."""
    print('=' * 50)
    print('  TEST DRIVE SIMULATOR - CONNECTED!')
    print('  Select a track and click "Autonomous Mode"')
    print('=' * 50)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    """Send steering and throttle commands to the simulator."""
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


def launch_simulator():
    """Auto-launch the TEST DRIVE SIMULATOR."""
    if not os.path.exists(SIMULATOR_PATH):
        print(f'[ERROR] TEST DRIVE SIMULATOR not found at: {SIMULATOR_PATH}')
        print('Please place the simulator in the beta_simulator_windows folder.')
        return None

    print(f'Launching TEST DRIVE SIMULATOR...')
    try:
        process = subprocess.Popen(
            [SIMULATOR_PATH],
            cwd=os.path.dirname(SIMULATOR_PATH)
        )
        print('TEST DRIVE SIMULATOR launched! Waiting for it to start...')
        time.sleep(3)  # Give simulator time to initialize
        return process
    except Exception as e:
        print(f'[ERROR] Failed to launch TEST DRIVE SIMULATOR: {e}')
        return None


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEST DRIVE SIMULATOR - Controller')
    parser.add_argument('--no-launch', action='store_true',
                        help='Skip auto-launching the TEST DRIVE SIMULATOR')
    args = parser.parse_args()

    print('=' * 50)
    print('  TEST DRIVE SIMULATOR')
    print('  Autonomous Driving Controller')
    print('=' * 50)

    # Load model
    print(f'\nLoading model from: {os.path.basename(MODEL_PATH)}')
    model = load_model(MODEL_PATH, compile=False)
    print('Model loaded successfully!\n')

    # Auto-launch simulator
    sim_process = None
    if not args.no_launch:
        sim_process = launch_simulator()
    else:
        print('Skipping TEST DRIVE SIMULATOR auto-launch (--no-launch flag)')

    # Free up port if still in use from a previous run
    try:
        import socket
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(1)
        result = test_sock.connect_ex(('127.0.0.1', PORT))
        test_sock.close()
        if result == 0:
            print(f'[INFO] Port {PORT} is in use. Cleaning up...')
            os.system(f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{PORT}\') do taskkill /F /PID %a 2>nul')
            time.sleep(1)
            print(f'[INFO] Port {PORT} freed up.')
    except Exception:
        pass

    # Start server
    print(f'\nStarting server on port {PORT}...')
    print('Waiting for TEST DRIVE SIMULATOR connection...\n')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', PORT)), app)