"""
Gaze Tracking Model Training Script
Trains a lightweight CNN on MPIIGaze dataset and exports to ONNX format
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from pathlib import Path

# Lightweight CNN for Gaze Estimation
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Input image: 60x36 -> after 3 pooling: 7x4
        self.fc1 = nn.Linear(128 * 7 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Output: (x, y) gaze direction
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 30x18
        x = self.pool(self.relu(self.conv2(x)))  # 15x9
        x = self.pool(self.relu(self.conv3(x)))  # 7x4
        
        x = x.view(-1, 128 * 7 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Dummy dataset for demonstration (replace with actual MPIIGaze data)
class DummyGazeDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy eye image (60x36 grayscale)
        image = np.random.rand(36, 60).astype(np.float32)
        
        # Generate dummy gaze direction (normalized coordinates)
        gaze = np.random.rand(2).astype(np.float32) * 2 - 1  # Range: [-1, 1]
        
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        gaze_tensor = torch.from_numpy(gaze)
        
        return image_tensor, gaze_tensor

def train_model(epochs=50, batch_size=32, learning_rate=0.001):
    """Train the gaze tracking model"""
    
    print("Initializing training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = GazeNet().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Data loader
    dataset = DummyGazeDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, gazes) in enumerate(dataloader):
            images = images.to(device)
            gazes = gazes.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gazes)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print("\nTraining completed!")
    return model

def export_to_onnx(model, output_path="gaze_model.onnx"):
    """Export the trained model to ONNX format"""
    
    print(f"\nExporting model to ONNX format: {output_path}")
    
    model.eval()
    
    # Dummy input (batch_size=1, channels=1, height=36, width=60)
    dummy_input = torch.randn(1, 1, 36, 60)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported successfully to: {output_path}")
    print(f"Input shape: (batch_size, 1, 36, 60)")
    print(f"Output shape: (batch_size, 2) - [x, y] gaze coordinates")

if __name__ == "__main__":
    print("=" * 60)
    print("Gaze Tracking Model Training")
    print("=" * 60)
    
    # Train the model
    trained_model = train_model(epochs=50, batch_size=32)
    
    # Export to ONNX
    export_to_onnx(trained_model, "gaze_model.onnx")
    
    print("\n" + "=" * 60)
    print("All done! ONNX model is ready for C++ inference.")
    print("=" * 60)
