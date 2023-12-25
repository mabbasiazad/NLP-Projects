#Note: implementing LeNet5 for NN model
import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define relevant variables for the ML task
BATCH_SIZE = 64
NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.1307,) , std=(0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='.\data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='.\data', train=True, transform=transform, download=True)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(16 * 5 *5, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # (B, C, W, H)
        x = self.layer1(x)
        x = self.layer2(x) # the outpout would be (B, C, W, H)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x 

if __name__ == "__main__":
    #======== test the model for dimensionality ========
    # model = Model()
    # input = torch.rand(1, 1, 32, 32)
    # output = model(input)
    # print(output.size())
    # print(output)

    model = Model().to(device)

    loss_fc = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    NUM_EPOCHS = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)
            loss = loss_fc(prediction, y)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch_idx % 400 == 0: 
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], loss: {loss.item():.4f}")
        
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in train_loader:
            x.to(device)
            y.to(device)
            output = model(x)
            prediction = torch.argmax(output, dim=1)
            total += y.size(0)
            correct += (y == prediction).sum().item()

    print("test accuracy %: ", (correct / total) * 100)

    

