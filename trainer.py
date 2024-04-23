# Import libraries
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as F
import cv2
from torch.utils.data import random_split
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=28):  # Default to 28 classes now
        super(SignLanguageCNN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.resnet = nn.Sequential(*(list(resnet.children())[1:-1]))
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def prepare_train_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Lambda(thresholding),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return dataset

def prepare_test_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Lambda(thresholding),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return dataset


def create_data_loader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=20):
    model.train()
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for image, y_true in train_loader:
            image, y_true = image.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(image)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * image.size(0)
            _, predicted_indices = torch.max(y_pred, 1)
            total_correct += (predicted_indices == y_true).sum().item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_accuracy = total_correct / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        print(f'Epoch {epoch+1}/{epochs} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        print()
    return train_losses, train_accuracies, test_losses, test_accuracies


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted_indices = torch.max(outputs, 1)
            correct += (predicted_indices == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy

def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Training Loss')
    plt.plot(epochs, test_losses, 'b-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'b-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def save_model(model, save_path='model.pth'):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



# def evaluate_model(model, test_loader, criterion):
#     """Evaluate the model performance on a test dataset."""
#     model.eval()
#     total_loss = 0
#     correct = 0
#     label_map = {i: chr(65 + i) for i in range(26)}  # Create label map

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             _, predicted_indices = torch.max(outputs, 1)
#             correct += (predicted_indices == labels).sum().item()
#             total_loss += loss.item() * images.size(0)
    
#             # Optionally print out predictions here
#             predicted_labels = [label_map[idx] for idx in predicted_indices.cpu().numpy()]
#             actual_labels = [label_map[idx] for idx in labels.cpu().numpy()]
#             print("Predicted:", predicted_labels)
#             print("Actual:", actual_labels)

#     avg_loss = total_loss / len(test_loader.dataset)
#     accuracy = 100 * correct / len(test_loader.dataset)
#     print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def thresholding(pil_image):
    # Convert PIL Image to a NumPy array for OpenCV processing
    image = np.array(pil_image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours and remove small noise
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:  # Adjust this value as needed
            cv2.drawContours(binary, [cnt], 0, 0, -1)
    
    # Convert binary image back to a PIL Image
    binary = cv2.bitwise_not(binary)
    return Image.fromarray(binary)




def create_data_loaders(full_dataset, train_size, batch_size):
    # Split the dataset into training and test sets
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create dataloaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_label_map(dataset):
    # This function creates a map from class indices to class names
    return {v: k for k, v in dataset.class_to_idx.items()}


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_dataloader_images(dataloader):
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))

    # Convert tensors to numpy arrays
    inputs = inputs.numpy()

    # We no longer apply any normalization or inverse normalization
    fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=8, squeeze=False)
    for idx, (ax, img, cls) in enumerate(zip(axes.flat, inputs, classes)):
        # Since img is now a single channel (grayscale), we use a colormap to display it
        ax.imshow(img.squeeze(), cmap='gray')  # img.squeeze() to remove channel dimension for display
        ax.axis('off')
        ax.set_title(f'Class: {cls}')

    plt.tight_layout()
    plt.show()


def main():
    num_classes = 28
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    model_path = 'sign_language_cnn_space_delete.pth'
    
    train_dataset = prepare_train_dataset('asl_dataset')
    test_dataset = prepare_test_dataset('final_test_dataset')
    
    # train_size = int(0.8 * len(full_dataset))
    # train_loader, test_loader = create_data_loaders(full_dataset, train_size, batch_size)
    
    train_loader = create_data_loader(train_dataset, batch_size)
    test_loader = create_data_loader(test_dataset, batch_size)

    label_map = get_label_map(train_dataset)
    print("Label map:", label_map)

    show_dataloader_images(train_loader)
    show_dataloader_images(test_loader)

    model = SignLanguageCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)
    
    save_model(model, model_path)
    
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)

if __name__ == "__main__":
    main()






