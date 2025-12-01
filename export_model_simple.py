"""
Simple model export using TorchScript (alternative to ONNX)
"""

import torch
import torch.nn as nn
import numpy as np

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

print("Creating and training simple gaze model...")

# Create model
model = GazeNet()
model.eval()

# Train on dummy data for a few iterations
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.train()
for i in range(100):
    x = torch.randn(4, 1, 36, 60)
    y = torch.randn(4, 2)
    
    output = model(x)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 20 == 0:
        print(f"Step {i+1}/100, Loss: {loss.item():.4f}")

print("\nTraining complete!")

# Export to TorchScript
model.eval()
example_input = torch.randn(1, 1, 36, 60)

print("\nExporting to TorchScript...")
traced_model = torch.jit.trace(model, example_input)
traced_model.save("gaze_model.pt")
print("✓ Model saved to gaze_model.pt")

# Also try ONNX export with legacy exporter
print("\nAttempting ONNX export...")
try:
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            "gaze_model.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
    print("✓ ONNX model saved to gaze_model.onnx")
except Exception as e:
    print(f"⚠ ONNX export failed: {e}")
    print("Using TorchScript model instead (gaze_model.pt)")

print("\n✓ Export complete!")
