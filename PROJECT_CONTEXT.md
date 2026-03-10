# Autonomous Driving Car - Capstone Project Context

## 📋 Project Overview

**Goal**: Build an autonomous driving car that can:
1. **Follow lanes** using computer vision and deep learning
2. **Avoid obstacles** without collisions
3. Work with the **Udacity Self-Driving Car Simulator**

**Student's Capstone Objective**: Demonstrate understanding of autonomous vehicle perception and control systems, with a focus on creating a transparent, educational implementation that showcases core ADAS (Advanced Driver Assistance Systems) principles.

---

## 🎯 Project Differentiators (For Evaluators)

Since this is a capstone project, here's how to position it against criticism like "how is this better than existing systems":

### Key Selling Points:
1. **Transparency over Black-Box**: Unlike Tesla/Waymo's proprietary systems, this is fully open-source and explainable
2. **Educational Platform**: Complete learning pipeline documented - perfect for academic study
3. **Research Extensibility**: Modular architecture allows experimentation and modification
4. **Accessible**: Runs on any laptop with a simulator (no $50K+ vehicle needed)
5. **Limitation-Aware**: Explicitly documents what works, what doesn't, and proposes extensions

### Addressing Simulator Limitations:
- Udacity simulator doesn't include pedestrians - **Proposed extension**: Integrate YOLO/SSD object detection
- Simulated environment only - **Discussion point**: Sim-to-real transfer challenges
- Limited scenarios - **Enhancement**: Add edge case handling (shadows, glare, fading lane markings)

---

## 🛠️ Setup Completed

### Source Repository
- **Original**: https://github.com/entbappy/Complete-Self-Driving-Car
- **Cloned and renamed to**: `c:\Users\User\Antigravity\Autonomous driving`

### Environment Setup
```
Platform: Windows 10/11
Python Version: 3.13.7
Virtual Environment: sdcar_env (recreated fresh on March 9, 2026)
```

### Dependencies Installed
All packages from `requirements.txt` successfully installed:
- **TensorFlow 2.21.0** - Deep learning framework
- **Keras 3.13.2** - Neural network API
- **OpenCV 4.13.0** - Computer vision library
- **NumPy 2.4.3** - Numerical computing
- **Matplotlib 3.10.8** - Visualization
- **Flask 3.1.3** - Web framework for server
- **python-socketio 5.16.1** - Real-time communication with simulator
- **eventlet 0.40.4** - Async networking
- **Pillow 12.1.1** - Image processing
- **h5py 3.14.0** - HDF5 file format (for model storage)

### Simulator
- **Udacity Self-Driving Car Simulator** downloaded separately by user
- Available at: https://github.com/udacity/self-driving-car-sim

---

## 📁 Project Structure

```
c:\Users\User\Antigravity\Autonomous driving\
├── sdcar_env\                    # Virtual environment with all dependencies
├── model\
│   └── model.h5                  # Pre-trained Keras model (~3.2MB)
├── Data\
│   └── driving_log.csv           # Training data log
├── 1.Finding_Lanes\
│   ├── lane.py                   # Lane detection algorithms
│   └── artifacts\                # Lane detection outputs
├── Notebooks\
│   ├── Behavioral Cloning.ipynb  # Main training notebook
│   └── Traffic Signs Classification.ipynb
├── Self_Driving_Car.ipynb        # Complete training pipeline notebook (~7.4MB)
├── drive.py                      # Script to connect to simulator and drive
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── issue.txt                     # Known issues
└── simulator-windows-64\         # Simulator reference folder
```

---

## 🔧 Key Components Explained

### 1. `drive.py` - The Brain
This script:
1. Loads the trained model (`model/model.h5`)
2. Creates a SocketIO server on port 4567
3. Receives camera images from the simulator
4. Preprocesses images and predicts steering angle
5. Sends steering + throttle commands back

**Preprocessing Pipeline**:
```python
def img_preprocess(img):
    img = img[60:135,:,:]           # Crop to remove sky and hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV colorspace
    img = cv2.GaussianBlur(img, (3, 3), 0)      # Reduce noise
    img = cv2.resize(img, (200, 66))             # Resize for NVIDIA architecture
    img = img/255                                # Normalize pixel values
    return img
```

### ⚠️ Fixes Applied (March 2026):
- **Keras 3.x Compatibility**: Added `compile=False` to `load_model()` since the old H5 model uses metrics/loss functions from Keras 2.x that can't be deserialized in Keras 3.x. This is fine since we only need inference.
- **Suppressed Deprecation Warnings**: Eventlet deprecation warnings suppressed for cleaner output.
- **Verbose=0 on predict()**: Suppressed per-prediction logging for cleaner terminal output.

### 2. `Self_Driving_Car.ipynb` - Training Pipeline
Contains:
- Data loading and augmentation
- CNN model architecture (based on NVIDIA's End-to-End Learning paper)
- Training with behavioral cloning
- Model evaluation

### 3. `model/model.h5` - Pre-trained Model
- Keras model trained on Lake Track data
- Uses behavioral cloning approach
- Predicts steering angle from camera image

---

## 🚀 How to Run

### Step 1: Activate Environment
```bash
cd "c:\Users\User\Antigravity\Autonomous driving"
sdcar_env\Scripts\activate
```

### Step 2: Start the Python Server
```bash
python drive.py
```

### Step 3: Launch Simulator
1. Open Udacity Self-Driving Car Simulator
2. Select a track (Lake Track recommended for pre-trained model)
3. Click "Autonomous Mode"

### Step 4: Watch the Car Drive!
The terminal will show real-time output:
```
steering_angle  throttle  speed
0.05            0.8       9.5
```

---

## 📊 Next Steps / Roadmap

### Immediate Tasks:
1. [/] **Test pre-trained model** with simulator (server running, awaiting simulator connection)
2. [ ] **Record training data** by driving manually in Training Mode
3. [ ] **Retrain model** with custom data if needed

### Enhancement Ideas:
1. [ ] **Add real-time dashboard** - Visualize model confidence, steering predictions
2. [ ] **Implement safety layer** - Emergency braking when uncertainty is high
3. [ ] **Model comparison study** - Test CNN vs. NVIDIA architecture vs. Transformers
4. [ ] **Night/weather augmentation** - Test robustness with augmented data
5. [ ] **Object detection module** - Integrate YOLO for obstacle/pedestrian detection
6. [ ] **Modular architecture** - Separate lane detection + obstacle avoidance

### Documentation for Capstone:
1. [ ] Create detailed project report
2. [ ] Prepare presentation slides
3. [ ] Document model architecture and training process
4. [ ] Record demo video of autonomous driving
5. [ ] Prepare answers for evaluator questions

---

## 💡 Technical Details for Future AI Context

### Architecture
- **Behavioral Cloning**: The model learns to mimic human driving behavior
- **CNN-based**: Uses convolutional neural networks for image understanding
- **End-to-End Learning**: Direct mapping from pixels to steering commands
- **Based on NVIDIA Paper**: "End to End Learning for Self-Driving Cars" (2016)

### Communication Protocol
- **SocketIO** for real-time bidirectional communication
- **Flask** as the web server backend
- **Port 4567** for simulator connection
- **Base64 encoded images** sent from simulator

### Model Input/Output
- **Input**: 66x200x3 image (YUV colorspace)
- **Output**: Single float value (steering angle, typically -1 to 1)

### Training Data Format
- `driving_log.csv` contains: center_image, left_image, right_image, steering, throttle, brake, speed
- Images stored in `IMG/` folder (created during training mode)

---

## ⚠️ Known Limitations

1. **No pedestrian detection** - Simulator doesn't include pedestrians
2. **Track-specific model** - Pre-trained model works best on Lake Track
3. **No night driving** - Model not trained on night conditions
4. **Simple obstacle avoidance** - Basic collision prevention only
5. **Simulation only** - Not tested on real hardware

---

## 📞 Commands Reference

```bash
# Activate environment
cd "c:\Users\User\Antigravity\Autonomous driving"
sdcar_env\Scripts\activate

# Run autonomous driving
python drive.py

# Check installed packages
pip list

# Deactivate environment
deactivate
```

---

*Last Updated: March 9, 2026*
*Project Location: c:\Users\User\Antigravity\Autonomous driving*
