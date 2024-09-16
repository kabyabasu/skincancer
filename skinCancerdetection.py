from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Set the device
device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def training_dataprocessing(location):
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure resizing before ToTensor
        transforms.ToTensor()
    ])

    dataset_train = ImageFolder(
        location, transform=train_transform
    )

    dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)

    return dataloader_train


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classification = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classification(x)
        return x


# Load data
train_data = training_dataprocessing('./skin_cancer/Train')

# Initialize the network and move it to the GPU
net = Net(num_classes=9).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    running_loss = 0
    for image, label in train_data:
        # Move data to the GPU and ensure it has the correct type
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = net(image)
        loss = criterion(outputs, label)  # Ensure both outputs and labels are on the same device
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_data)
    print(f"Epoch {epoch + 1} : Loss {epoch_loss:.4f}")

# Save the trained model
torch.save(net.state_dict(), 'skincancermodel.pth')
