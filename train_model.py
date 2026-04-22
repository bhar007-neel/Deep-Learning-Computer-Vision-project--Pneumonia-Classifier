import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from PIL import Image

# --------------------------------------------------
# STEP 1: Choose device
# --------------------------------------------------
# If a GPU is available, use it because it trains faster.
# If not, PyTorch will use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# STEP 2: Create custom dataset class
# --------------------------------------------------
# This class helps PyTorch understand:
# 1. Where the images are stored
# 2. What label each image has
# 3. How to load one image when needed
class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): path to dataset folder like data/train
            transform (callable, optional): preprocessing steps for images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # The dataset has 2 folders:
        # NORMAL -> label 0
        # PNEUMONIA -> label 1
        for label in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(root_dir, label)

            # Loop through every file inside the class folder
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)

                # Save full image path
                self.image_paths.append(img_path)

                # Convert text label into a numeric label
                if label == "NORMAL":
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __len__(self):
        """
        Returns:
            int: total number of images in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads one image and its label using the given index.

        Args:
            idx (int): index of the image

        Returns:
            image (Tensor): processed image
            label (int): class label (0 or 1)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image using PIL and convert it to RGB
        # RGB means 3 color channels: Red, Green, Blue
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# STEP 3: Define image transformations
# --------------------------------------------------
# Neural networks cannot use raw images directly.
# So we preprocess them before sending them into the model.
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize all images to 224x224 for ResNet
    transforms.ToTensor(),           # Convert image into PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
        std=[0.229, 0.224, 0.225]    # Standard ImageNet standard deviation
    )
])


# --------------------------------------------------
# STEP 4: Create datasets
# --------------------------------------------------
# We create separate datasets for training, validation, and testing.
train_dataset = PneumoniaDataset(root_dir="data/train", transform=transform)
val_dataset = PneumoniaDataset(root_dir="data/val", transform=transform)
test_dataset = PneumoniaDataset(root_dir="data/test", transform=transform)


# --------------------------------------------------
# STEP 5: Create dataloaders
# --------------------------------------------------
# DataLoader loads images in small batches instead of all at once.
# This helps with memory and makes training efficient.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --------------------------------------------------
# STEP 6: Load pretrained ResNet18 model
# --------------------------------------------------
# We use transfer learning:
# The model already learned features from ImageNet,
# and now we adjust it for our pneumonia classification task.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace the final fully connected layer
# because our problem has only 2 classes:
# 0 = NORMAL
# 1 = PNEUMONIA
model.fc = nn.Linear(model.fc.in_features, 2)

# Move model to selected device (GPU or CPU)
model = model.to(device)


# --------------------------------------------------
# STEP 7: Define loss function and optimizer
# --------------------------------------------------
# CrossEntropyLoss is commonly used for multi-class classification
# and also works for binary classification with 2 output neurons.
criterion = nn.CrossEntropyLoss()

# Adam optimizer updates the model weights during training
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --------------------------------------------------
# STEP 8: Set number of epochs
# --------------------------------------------------
# One epoch means the model sees the full training dataset once.
num_epochs = 10


# --------------------------------------------------
# STEP 9: Training loop
# --------------------------------------------------
# This is where the model learns from the training data.
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Loop through training batches
    for images, labels in train_loader:
        # Move images and labels to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients from previous batch
        optimizer.zero_grad()

        # Forward pass:
        # Send images into the model and get predictions
        outputs = model(images)

        # Compute loss:
        # Compare model predictions with actual labels
        loss = criterion(outputs, labels)

        # Backward pass:
        # Calculate gradients for all trainable parameters
        loss.backward()

        # Update weights using optimizer
        optimizer.step()

        # Add current batch loss to total loss
        running_loss += loss.item()

    # Print average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# --------------------------------------------------
# STEP 10: Validation
# --------------------------------------------------
# Validation is used to check how well the model performs
# on unseen validation data after training.
model.eval()  # Set model to evaluation mode

val_labels = []
val_preds = []

# No gradient calculation is needed during evaluation
with torch.no_grad():
    for images, labels in val_loader:
        # Move validation batch to device
        images = images.to(device)
        labels = labels.to(device)

        # Get model outputs
        outputs = model(images)

        # torch.max returns:
        # 1. maximum value
        # 2. index of maximum value
        # We only need the predicted class index
        _, preds = torch.max(outputs, 1)

        # Move tensors back to CPU and convert to numpy
        # so sklearn can use them
        val_labels.extend(labels.cpu().numpy())
        val_preds.extend(preds.cpu().numpy())

# Calculate validation accuracy
val_accuracy = accuracy_score(val_labels, val_preds)
print("Validation accuracy:", val_accuracy)


# --------------------------------------------------
# STEP 11: Testing
# --------------------------------------------------
# Test accuracy is the final performance measure
# on completely unseen data.
model.eval()  # Keep model in evaluation mode

test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        # Move test batch to device
        images = images.to(device)
        labels = labels.to(device)

        # Get predictions from the model
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Store true labels and predicted labels
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

# Calculate test accuracy
test_accuracy = accuracy_score(test_labels, test_preds)
print("Test accuracy:", test_accuracy)


# --------------------------------------------------
# STEP 12: Save trained model
# --------------------------------------------------
# This saves only the learned weights of the model.
# You can load them later without training again.
torch.save(model.state_dict(), "pneumonia_classifier.pth")
print("Model saved successfully as pneumonia_classifier.pth")