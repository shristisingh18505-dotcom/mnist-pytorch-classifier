"""
MNIST Handwritten-Digit Classifier
Beginner PyTorch project

Deliverables:
1. Simple baseline: linear classifier
2. Small neural network: 2-layer MLP
3. 5 sample predictions
4. Simple architecture sketch
5. Explanation of pixels -> output class

Run:
    python mnist_classifier.py

Requirements:
    pip install torch torchvision matplotlib
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS_BASELINE = 3
EPOCHS_NN = 5
LEARNING_RATE = 0.1
NN_LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------------------------------------
# 2. LOAD MNIST DATASET
# ------------------------------------------------------------
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print("Training images:", len(training_data))
print("Test images:", len(test_data))
print("One image shape:", training_data[0][0].shape)
print("Pixel values are converted to tensors by ToTensor().")


# ------------------------------------------------------------
# 3. EVALUATION FUNCTION
# ------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            predictions = output.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return 100 * correct / total


# ------------------------------------------------------------
# 4. SIMPLE BASELINE
# ------------------------------------------------------------
# A linear classifier is our baseline.
# It receives 28x28 = 784 pixel values and directly predicts 10 classes.
class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


baseline = LinearBaseline().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(baseline.parameters(), lr=LEARNING_RATE)

print("\n--- Training baseline ---")

for epoch in range(EPOCHS_BASELINE):
    baseline.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        predictions = baseline(X)
        loss = loss_fn(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = evaluate(baseline, test_loader)
    print(f"Baseline epoch {epoch + 1}/{EPOCHS_BASELINE} - "
          f"loss: {loss.item():.4f} - test accuracy: {accuracy:.2f}%")

baseline_accuracy = evaluate(baseline, test_loader)


# ------------------------------------------------------------
# 5. SMALL NEURAL NETWORK
# ------------------------------------------------------------
# Architecture:
# 784 input pixels
#       |
#   Linear 128
#       |
#      ReLU
#       |
#   Linear 64
#       |
#      ReLU
#       |
#   Linear 10
#       |
#  predicted digit
class SmallNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


model = SmallNeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE)

print("\n--- Training neural network ---")

for epoch in range(EPOCHS_NN):
    model.train()
    running_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        predictions = model(X)

        # Measure error
        loss = loss_fn(predictions, y)

        # Backpropagation + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    accuracy = evaluate(model, test_loader)
    average_loss = running_loss / len(train_loader)

    print(f"NN epoch {epoch + 1}/{EPOCHS_NN} - "
          f"loss: {average_loss:.4f} - test accuracy: {accuracy:.2f}%")

nn_accuracy = evaluate(model, test_loader)

print("\n================ RESULTS ================")
print(f"Linear baseline accuracy: {baseline_accuracy:.2f}%")
print(f"Neural network accuracy:  {nn_accuracy:.2f}%")
print("==========================================")


# ------------------------------------------------------------
# 6. SHOW 5 SAMPLE PREDICTIONS
# ------------------------------------------------------------
model.eval()

# Pick five examples from the test set.
import random
sample_indices = torch.randperm(len(test_data))[:5].tolist()

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

with torch.no_grad():
    for position, index in enumerate(sample_indices):
        image, true_label = test_data[index]

        output = model(image.unsqueeze(0).to(device))
        predicted_label = output.argmax(dim=1).item()

        axes[position].imshow(image.squeeze(), cmap="gray")
        axes[position].set_title(
            f"True: {true_label}\nPred: {predicted_label}"
        )
        axes[position].axis("off")

        print(
            f"Example {position + 1}: "
            f"True digit = {true_label}, "
            f"Predicted digit = {predicted_label}"
        )

plt.tight_layout()
plt.savefig("five_predictions.png", dpi=150)
plt.show()


# ------------------------------------------------------------
# 7. SAVE TRAINED MODEL
# ------------------------------------------------------------
torch.save(model.state_dict(), "mnist_small_nn.pth")
print("\nSaved trained model as mnist_small_nn.pth")


# ------------------------------------------------------------
# 8. SIMPLE ARCHITECTURE SKETCH
# ------------------------------------------------------------
architecture = """
MNIST IMAGE
28 x 28 pixels
      |
      v
Flatten
784 numbers
      |
      v
Linear Layer
784 -> 128
      |
      v
    ReLU
      |
      v
Linear Layer
128 -> 64
      |
      v
    ReLU
      |
      v
Linear Layer
64 -> 10
      |
      v
10 output scores
      |
      v
Highest score = predicted digit
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
"""

print(architecture)
