import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(0.4, 1.3)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_shadow(image):
    h, w = image.shape[:2]
    x1, x2 = np.random.randint(0, w, 2)
    if x1 > x2:
        x1, x2 = x2, x1
    shadow_mask = np.ones_like(image, dtype=np.float32)
    shadow_mask[:, x1:x2, :] = np.random.uniform(0.3, 0.7)
    return np.clip(image * shadow_mask, 0, 255).astype(np.uint8)

def random_translate(image, x_range=100, y_range=10):
    dx = np.random.uniform(-x_range, x_range)
    dy = np.random.uniform(-y_range, y_range)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image

def main():
    # Find a sample image
    img_dir = "Data/IMG/"
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not images:
        print("No images found in Data/IMG/")
        return
    
    # Take a random image or just the first one
    img_path = images[len(images)//2]
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create augmentations
    flipped = cv2.flip(original, 1)
    translated = random_translate(original)
    brightened = random_brightness(original)
    shadowed = random_shadow(original)
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Data Augmentation Process for Training Generalization", fontsize=16)
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(flipped)
    axes[1].set_title("Horizontal Flip")
    axes[1].axis('off')
    
    axes[2].imshow(translated)
    axes[2].set_title("Random Translation")
    axes[2].axis('off')
    
    axes[3].imshow(brightened)
    axes[3].set_title("Random Brightness")
    axes[3].axis('off')
    
    axes[4].imshow(shadowed)
    axes[4].set_title("Random Shadow")
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_sample.png', dpi=300, bbox_inches='tight')
    print("Saved augmentation_sample.png")

if __name__ == "__main__":
    main()
