import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Create a recurrent neural network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # Take the last time step output
        out = self.fc(out)
        return out


# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((input_size, input_size)),  # Resize to the original MNIST image size
])

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
# Train network
# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Resize and reshape the data tensor
        data = data.view(data.size(0), sequence_length, -1)

        # Forward
        scores = model(data)

        # Calculate loss
        loss = criterion(scores, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")




# Check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    
    model.eval()
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # Resize and reshape the data tensor
            data = data.view(data.size(0), sequence_length, -1)
            
            # Forward pass
            scores = model(data)
            _, predictions = scores.max(1)
            
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        
        accuracy = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy {accuracy * 100:.2f}')
    
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
