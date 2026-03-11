# Vision-Driven Autonomous Navigation Using Deep Neural Networks in a Simulated Environment

A self-driving car system that uses deep learning (NVIDIA's CNN architecture) to predict steering angles from camera images, enabling autonomous navigation in the Udacity Self-Driving Car Simulator.

![Self-Driving Car](https://cdn.dribbble.com/users/1815/screenshots/2589016/car_dr.gif)


## Features

- **NVIDIA CNN Architecture** — End-to-end learning from pixels to steering commands
- **Advanced Data Augmentation** — Random brightness, shadows, translations, and flips for robust generalization
- **Auto-Launch Simulator** — `drive.py` automatically opens the TEST DRIVE SIMULATOR
- **Real-Time Telemetry** — Live steering, throttle, and speed output in terminal


## Setup

### Step 1: Create & activate virtual environment
```bash
python -m venv sdcar_env
sdcar_env\Scripts\activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
pip install pandas scikit-learn
pip install python-socketio==4.6.1 python-engineio==3.13.2
```

### Step 3: Collect training data
- Open the simulator in **Training Mode**
- Drive 3-5 smooth laps (both directions)
- Data saves to `Data/driving_log.csv` and `Data/IMG/`

### Step 4: Train the model
```bash
python train_model.py --epochs 15 --batch 100 --steps 300
```

### Step 5: Run autonomous driving
```bash
python drive.py
```
Select **Autonomous Mode** in the simulator and watch the car drive itself!


## Project Structure

```
├── train_model.py          # Training script with augmentation pipeline
├── drive.py                # Autonomous driving controller (TEST DRIVE SIMULATOR)
├── collect_data.py         # Data collection helper
├── Data/
│   ├── driving_log.csv     # Training data log
│   └── IMG/                # Camera images (center, left, right)
├── model/
│   └── model.h5            # Trained Keras model
├── beta_simulator_windows/ # Udacity Self-Driving Car Simulator
├── requirements.txt        # Python dependencies
└── README.md
```


## Author

**Aaron Shoby John**
Email: aaronshoby319@gmail.com
GitHub: [@AaronShoby](https://github.com/AaronShoby)
