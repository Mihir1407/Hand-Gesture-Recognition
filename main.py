# import statements
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import numpy as np
import cv2
import mediapipe as mp

# Setup device for Torch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def detect_and_crop(image):
    """Detect hands using MediaPipe and crop the image around the detected hand."""
    # Convert PIL image to cv2 format
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Process the image and detect hands
    results = hands.process(image_cv2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Assuming only one hand per image for simplicity
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.width
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.width
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.height
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.height
            
            # Expand the bounding box slightly
            x_min = max(0, x_min - 0.05 * image.width)
            x_max = min(image.width, x_max + 0.05 * image.width)
            y_min = max(0, y_min - 0.05 * image.height)
            y_max = min(image.height, y_max + 0.05 * image.height)
            
            # Crop the image
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            return cropped_image

    return image  # Return the original if no hand is detected

# Custom dataset class remains unchanged
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, base_save_path=''):
        self.dataframe = dataframe
        self.transform = transform
        self.base_save_path = base_save_path  # Base path to save cropped images

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency

        # Detect hand and crop the image
        cropped_image = detect_and_crop(image)

        # Save the cropped image following the original folder structure
        if self.base_save_path:
            rel_path = os.path.relpath(img_path, start=os.path.dirname(os.path.dirname(img_path)))  # relative path from the dataset folder
            save_path = os.path.join(self.base_save_path, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cropped_image.save(save_path)

        if self.transform:
            image = self.transform(cropped_image)

        return image, label

# useful functions (no changes needed here)
def load_images_from_folder(folder, is_test=False):
    """Load images and labels from given folder, handling test differently"""
    images = []
    labels = []
    if is_test:
        for img in os.listdir(folder):
            images.append(os.path.join(folder, img))
            labels.append(img.split('_')[0])  # Assumes format like A_test.jpg for test images
    else:
        for label in os.listdir(folder):
            path = os.path.join(folder, label)
            for img in os.listdir(path):
                images.append(os.path.join(path, img))
                labels.append(label)
    return images, labels

def create_dataframe(images, labels):
    """Create a pandas DataFrame with image paths and labels"""
    data = {'Image': images, 'Label': labels}
    return pd.DataFrame(data)

def display_images(dataset, title="Images"):
    """Display images from a dataset, reversing normalization."""
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        image, label = dataset[i]  # Get processed image and label from dataset
        
        # Reverse normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = image * std + mean  # Denormalize

        # Ensure image is within 0-1 range
        image = torch.clamp(image, 0, 1)

        # Convert tensor back to PIL image for displaying
        image = transforms.functional.to_pil_image(image)
        ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def get_transform():
    """Get the transformations for the images"""
    return transforms.Compose([
        transforms.Resize((128, 128)),  # Resize post cropping
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main(argv):
    """Main function to handle data loading, processing, and display"""
    train_folder = 'ASL_Alphabet_Dataset/asl_alphabet_train'
    test_folder = 'ASL_Alphabet_Dataset/asl_alphabet_test'
    cropped_train_folder = 'cropped_asl_alphabet_train'
    cropped_test_folder = 'cropped_asl_alphabet_test'

    # Load training images and labels
    train_images, train_labels = load_images_from_folder(train_folder)
    train_df = create_dataframe(train_images, train_labels)
    print("Training DataFrame Head:")
    print(train_df.head())

    # Transformations
    transform = get_transform()

    # Create datasets
    train_dataset = ImageDataset(train_df, transform=transform, base_save_path=cropped_train_folder)

    # Display training images (now cropped)
    display_images(train_dataset, title="Training Images (Cropped)")

    # Load test images and labels
    test_images, test_labels = load_images_from_folder(test_folder, is_test=True)
    test_df = create_dataframe(test_images, test_labels)
    print("Testing DataFrame Head:")
    print(test_df.head())

    # Create test dataset using test_df
    test_dataset = ImageDataset(test_df, transform=transform, base_save_path=cropped_test_folder)

    # Display test images (now cropped)
    display_images(test_dataset, title="Test Images (Cropped)")

    # Create DataLoaders (if needed later)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == "__main__":
    import sys
    main(sys.argv)

