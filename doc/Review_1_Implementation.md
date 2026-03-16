# 4. Implementation Details

## 4.1. Project Plan & Timeline
The development of the autonomous driving system was divided into several key phases:

1. **Phase 1: Environment Setup & Data Collection**
   - Configured the development environment using Python, TensorFlow/Keras, and OpenCV.
   - Installed and integrated the Udacity Self-Driving Car Simulator (TEST DRIVE SIMULATOR).
   - Manually drove the vehicle in the simulator to record telemetry data (steering angles, throttle, speed) and captured center, left, and right camera images.

2. **Phase 2: Data Preprocessing & Augmentation**
   - Cleaned the dataset by extracting absolute paths to images.
   - Implemented an advanced data augmentation pipeline:
     - **Random Brightness:** To simulate varying lighting conditions.
     - **Random Shadows:** To help the model handle shaded sections of the track.
     - **Translations:** To simulate different road positions.
     - **Horizontal Flips:** To eliminate left-turn biases and double the dataset size.

3. **Phase 3: Model Architecture & Training**
   - Built a Convolutional Neural Network (CNN) based on NVIDIA's End-to-End Learning Architecture.
   - Trained the model using Behavioral Cloning on the collected telemetry data to minimize Mean Squared Error (MSE) between predicted and actual steering angles.

4. **Phase 4: Autonomous Navigation & Testing**
   - Integrated the trained model (`model.h5`) with a Flask server running Socket.IO.
   - Re-established real-time bi-directional telemetry connection (`drive.py`) with the simulator.
   - Successfully validated the model on Track 1 with independent steering and speed adjustments.

## 4.2. Sample Code

### 4.2.1. Model Architecture (NVIDIA End-to-End Learning)
*This code defines the core neural network that processes camera images and predicts the steering angle.*
```python
def create_model():
    model = Sequential()
    # Normalize images
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))
    
    # Convolutional Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1)) # Output: Steering Angle
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
    return model
```

### 4.2.2. Real-time Telemetry Control
*This snippet receives real-time simulator data, processes the image, and sends the steering prediction back to the car.*
```python
@sio.on('telemetry')
def telemetry(sid, data):
    if data is None:
        return
    
    # Extract current speed and image
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    
    # Preprocess and predict
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image, verbose=0)[0][0])
    
    # Calculate throttle and send controls
    throttle = 1.0 - speed / SPEED_LIMIT
    send_control(steering_angle, throttle)
```

## 4.3. Sample Screenshots

*(Note: Replace the bracketed text below with the actual screenshots from your project folder.)*

1. **Simulator in Training Mode:**
   `[Insert screenshot showing the car being manually driven for data collection]`
   *Caption: Collecting manual driving data (steering, throttle, camera feeds) using the TEST DRIVE SIMULATOR.*

2. **Data Augmentation Process:**
   `[Insert screenshot or plot of an original image vs translated/shadowed image]`
   *Caption: Sample images showing advanced data augmentations applied during training to improve track generalization.*

3. **Training Loss Chart:**
   `[Insert screenshot of 'model/training_history.png' if available, or the command line output showing final validation loss]`
   *Caption: Training and Validation Loss minimizing over 15 epochs, demonstrating successful model convergence without overfitting.*

4. **Autonomous Mode in Action:**
   `[Insert screenshot showing the car driving by itself with 'Mode: Autonomous' on the screen]`
   *Caption: The trained CNN successfully navigating Track 1 autonomously, predicting real-time steering commands via WebSocket telemetry.*
