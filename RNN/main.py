import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28
seq_len = 28
num_layers = 2
hidden_size = 256
num_classes = 10
lr = 1e-3
batch_size = 64
num_epochs = 2


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size= hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn. LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*seq_len, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        #out, _ = self.LSTM(x, (h0, c0))
        out = out.reshape(out.shape[0], -1) #out = self.fc(out[:, -1,:]) for last hidden layer
        out = self.fc(out)
        return out

train_data = datasets.MNIST(root = "datasets/", train = True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root = "datasets/",train= False, download=True, transform=transforms.ToTensor())

trainloader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True)
testloader=  DataLoader(dataset=test_data, batch_size = batch_size, shuffle = True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
crit = nn.CrossEntropyLoss()
opti = optim.Adam(model.parameters(), lr = lr)

for i in range(num_epochs):
    for batch_idx, (x,y) in enumerate(trainloader):
        x = x.to(device).squeeze(1)
        y = y.to(device)
        opti.zero_grad()

        scores = model(x)
        loss = crit(scores, y)

        loss.backward()
        opti.step()

def accuracy(loader, model):
    if loader.dataset.train:
        print("Accuracy on train dataset")
    else:
        print("Accuracy on test dataset")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        print(f"Got {num_correct} / {num_samples} with acc {num_correct/num_samples * 100}")
    model.train()

accuracy(trainloader, model)
accuracy(testloader, model)





