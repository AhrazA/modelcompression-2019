import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classifier:
    def __init__(self, model, device, train_loader, test_loader):
        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._model = model.to(device)
        self._train_loader = train_loader
        self._test_loader = test_loader
    
    @property
    def model(self):
        return self._model
    
    def train(self, log_interval, optimizer, epoch, loss_fn):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self._train_loader):
            data, target = data.to(self._device), target.to(self._device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self._train_loader.dataset),
                    100. * batch_idx / len(self._train_loader), loss.item()))
    
    def test(self, loss_fn):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self._test_loader:
                data, target = data.to(self._device), target.to(self._device)
                output = self.model(data)
                test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        final_loss = test_loss / len(self._test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            final_loss, correct, len(self._test_loader.dataset),
            100. * correct / len(self._test_loader.dataset)))
        
        return correct / len(self._test_loader.dataset)
        
    
