import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from trainer import SignLanguageCNN


def load_model(model_path, device):
        num_classes = 28  # Number of classes (A-Z, 'delete', 'space')
        model = SignLanguageCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

def thresholding(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            cv2.drawContours(binary, [cnt], 0, 0, -1)
    binary = cv2.bitwise_not(binary)
    cv2.imshow("Thresholded Image", binary)
    return Image.fromarray(binary)


def getTransform(size):
    return transforms.Compose([
        transforms.Lambda(thresholding),
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def detect_and_crop_hand(frame, mp_hands):
        """Detect hands using MediaPipe and crop the image around the detected hand."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Assuming only one hand per image for simplicity
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                x_min, x_max = max(0, x_min - 50), min(frame.shape[1], x_max + 50)
                y_min, y_max = max(0, y_min - 50), min(frame.shape[0], y_max + 50)
                return frame[int(y_min):int(y_max), int(x_min):int(x_max)], True
        return frame, False