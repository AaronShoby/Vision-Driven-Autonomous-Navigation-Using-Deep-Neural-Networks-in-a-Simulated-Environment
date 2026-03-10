import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'Data')
IMG_DIR = os.path.join(DATA_DIR, 'IMG')
LOG_FILE = os.path.join(DATA_DIR, 'driving_log.csv')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

# Steering correction for left/right cameras
STEERING_CORRECTION = 0.2


# ============================================================
# Image Preprocessing (must match drive.py exactly)
# ============================================================
def img_preprocess(img):
    """
    Preprocess camera image for the CNN model.
    Based on NVIDIA's End-to-End Learning architecture:
    1. Crop - Remove sky (top 60px) and hood (bottom 25px)
    2. Convert to YUV colorspace (better for lane detection)
    3. Gaussian blur to reduce noise
    4. Resize to model input size (200x66)
    5. Normalize pixel values to [0, 1]
    """
    img = img[60:135, :, :]                        # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)     # Color space
    img = cv2.GaussianBlur(img, (3, 3), 0)         # Blur
    img = cv2.resize(img, (200, 66))               # Resize
    img = img / 255.0                              # Normalize
    return img


# ============================================================
# Data Loading
# ============================================================
def load_data():
    """Load driving log and prepare image paths + steering angles."""
    print(f'Loading data from: {LOG_FILE}')

    if not os.path.exists(LOG_FILE):
        print(f'\n[ERROR] No driving_log.csv found at:\n  {LOG_FILE}')
        print('Please collect training data first using: python collect_data.py')
        sys.exit(1)

    # Load CSV - columns: center, left, right, steering, throttle, brake, speed
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(LOG_FILE, names=columns, header=None)

    # Clean whitespace from path columns
    for col in ['center', 'left', 'right']:
        data[col] = data[col].str.strip()

    print(f'  Total entries: {len(data)}')
    print(f'  Steering range: [{data["steering"].min():.4f}, {data["steering"].max():.4f}]')
    print(f'  Mean steering: {data["steering"].mean():.4f}')

    # Build lists of image paths and steering angles
    # Use center, left, and right cameras with steering correction
    image_paths = []
    steerings = []

    for _, row in data.iterrows():
        steering = float(row['steering'])

        # Center camera
        center_path = row['center']
        if not os.path.isabs(center_path):
            center_path = os.path.join(DATA_DIR, center_path)
        image_paths.append(center_path)
        steerings.append(steering)

        # Left camera (steer right to correct)
        left_path = row['left']
        if not os.path.isabs(left_path):
            left_path = os.path.join(DATA_DIR, left_path)
        image_paths.append(left_path)
        steerings.append(steering + STEERING_CORRECTION)

        # Right camera (steer left to correct)
        right_path = row['right']
        if not os.path.isabs(right_path):
            right_path = os.path.join(DATA_DIR, right_path)
        image_paths.append(right_path)
        steerings.append(steering - STEERING_CORRECTION)

    image_paths = np.array(image_paths)
    steerings = np.array(steerings)

    print(f'  Total samples (with L/R cameras): {len(image_paths)}')
    return image_paths, steerings


# ============================================================
# Data Augmentation & Batch Generator
# ============================================================
def random_brightness(image):
    """Randomly adjust brightness to simulate different lighting conditions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(0.4, 1.3)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_shadow(image):
    """Add random shadow to simulate shadows on the road (helps with Track 2)."""
    h, w = image.shape[:2]
    # Random shadow region using a vertical strip
    x1, x2 = np.random.randint(0, w, 2)
    if x1 > x2:
        x1, x2 = x2, x1
    shadow_mask = np.ones_like(image, dtype=np.float32)
    shadow_mask[:, x1:x2, :] = np.random.uniform(0.3, 0.7)
    return np.clip(image * shadow_mask, 0, 255).astype(np.uint8)


def random_translate(image, steering, x_range=100, y_range=10):
    """Randomly shift image horizontally/vertically to simulate different road positions."""
    dx = np.random.uniform(-x_range, x_range)
    dy = np.random.uniform(-y_range, y_range)
    # Adjust steering proportionally to horizontal shift
    steering += dx * 0.002
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image, steering


def random_augment(image_path, steering):
    """Load and apply random augmentations to a single image."""
    image = cv2.imread(image_path)
    if image is None:
        # Try alternative path resolution
        alt_path = os.path.join(IMG_DIR, os.path.basename(image_path))
        image = cv2.imread(alt_path)
    if image is None:
        return None, steering

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        steering = -steering

    # Random translation (shift)
    if np.random.rand() > 0.5:
        image, steering = random_translate(image, steering)

    # Random brightness
    if np.random.rand() > 0.5:
        image = random_brightness(image)

    # Random shadow
    if np.random.rand() > 0.5:
        image = random_shadow(image)

    return image, steering


def batch_generator(image_paths, steerings, batch_size, is_training=True):
    """
    Generator that yields batches of preprocessed images and steering angles.
    Uses augmentation during training for better generalization.
    """
    while True:
        batch_images = []
        batch_steerings = []

        for _ in range(batch_size):
            idx = np.random.randint(len(image_paths))

            if is_training:
                image, steering = random_augment(image_paths[idx], steerings[idx])
            else:
                image = cv2.imread(image_paths[idx])
                if image is None:
                    alt_path = os.path.join(IMG_DIR, os.path.basename(image_paths[idx]))
                    image = cv2.imread(alt_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                steering = steerings[idx]

            if image is not None:
                image = img_preprocess(image)
                batch_images.append(image)
                batch_steerings.append(steering)

        yield np.array(batch_images), np.array(batch_steerings)


# ============================================================
# NVIDIA Model Architecture
# ============================================================
def nvidia_model():
    """
    NVIDIA End-to-End Learning CNN for self-driving cars.

    Architecture:
        - 5 Convolutional layers (24, 36, 48, 64, 64 filters)
        - 1 Dropout layer for regularization
        - 4 Fully connected layers (100, 50, 10, 1)
        - ELU activation throughout
        - Output: Single steering angle value

    Paper: "End to End Learning for Self-Driving Cars" (Bojarski et al., 2016)
    """
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Flatten + Dropout
    model.add(Flatten())
    model.add(Dropout(0.5))

    # Fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))  # Output: steering angle

    # Compile
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

    return model


# ============================================================
# Training Visualization
# ============================================================
def plot_training_history(history, save_path=None):
    """Plot and optionally save training/validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Model Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Training plot saved to: {save_path}')
    else:
        plt.show()


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train Self-Driving Car Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=100, help='Batch size')
    parser.add_argument('--steps', type=int, default=300, help='Steps per epoch')
    parser.add_argument('--val-steps', type=int, default=200, help='Validation steps per epoch')
    args = parser.parse_args()

    print('=' * 60)
    print('  MODEL TRAINING - SELF-DRIVING CAR')
    print('  NVIDIA End-to-End Learning Architecture')
    print('=' * 60)

    # 1. Load data
    print('\n[Step 1/5] Loading training data...')
    image_paths, steerings = load_data()

    if len(image_paths) < 100:
        print(f'\n[WARNING] Very few samples ({len(image_paths)})')
        print('Consider collecting more data for better results.')
        print('Recommended: 3-5 laps of driving in Training Mode.')

    # 2. Train/Validation split
    print('\n[Step 2/5] Splitting data (80% train, 20% validation)...')
    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=42
    )
    print(f'  Training samples: {len(X_train)}')
    print(f'  Validation samples: {len(X_valid)}')

    # 3. Build model
    print('\n[Step 3/5] Building NVIDIA CNN model...')
    model = nvidia_model()
    model.summary()

    # 4. Train
    print(f'\n[Step 4/5] Training model ({args.epochs} epochs)...')
    print(f'  Batch size: {args.batch}')
    print(f'  Steps per epoch: {args.steps}')
    print(f'  Validation steps: {args.val_steps}')

    history = model.fit(
        batch_generator(X_train, y_train, args.batch, is_training=True),
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        validation_data=batch_generator(X_valid, y_valid, args.batch, is_training=False),
        validation_steps=args.val_steps,
        verbose=1
    )

    # 5. Save model
    print('\n[Step 5/5] Saving model...')
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f'  Model saved to: {MODEL_PATH}')

    # Plot training history
    plot_path = os.path.join(MODEL_DIR, 'training_history.png')
    plot_training_history(history, save_path=plot_path)

    # Summary
    print('\n' + '=' * 60)
    print('  TRAINING COMPLETE!')
    print('=' * 60)
    print(f'  Final training loss: {history.history["loss"][-1]:.6f}')
    print(f'  Final validation loss: {history.history["val_loss"][-1]:.6f}')
    print(f'\n  Model saved to: {MODEL_PATH}')
    print(f'  Training plot: {plot_path}')
    print(f'\n  To test the model:')
    print(f'    python drive.py')
    print(f'  Then select Autonomous Mode in the simulator.')


if __name__ == '__main__':
    main()
