import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class Classifier:
    def __init__(self, model, device, train_loader, test_loader):
        self._device = device
        self._model = model.to(device)
        self._train_loader = train_loader
        self._test_loader = test_loader
    
    @property
    def model(self):
        return self._model
    
    def train(self, log_interval, optimizer, epochs, loss_fn):
        self.model.train()
        
        best_acc = -1
        best_weights = None

        for epoch in range(epochs):
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
            
            with torch.no_grad():
                acc = self.test(loss_fn)

                if acc > best_acc:
                    best_acc = acc
                    best_weights = copy.deepcopy(self.model.state_dict())
        
        return best_acc, best_weights
    
    def test(self, loss_fn, multiple_pred=False, reps=5):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            all_outputs = []

            for data, target in self._test_loader:
                data, target = data.to(self._device), target.to(self._device)

                if multiple_pred:
                    outputs = None
                    for i in range(reps):
                        output = self.model(data)
                        if outputs is None: outputs = output
                        else: outputs = torch.cat((outputs, self.model(data)))
                    
                    output = torch.empty_like(output)
                    for i, repeated_preds in enumerate(torch.split(outputs, output.shape[0])):
                        output[i] = torch.mean(repeated_preds, 0)
                    
                    all_outputs.append(output)
                                    
                else:
                    output = self.model(data)
                
                test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        final_loss = test_loss / len(self._test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            final_loss, correct, len(self._test_loader.dataset),
            100. * correct / len(self._test_loader.dataset)))
        
        if multiple_pred:
            return correct / len(self._test_loader.dataset), all_outputs
        
        return correct / len(self._test_loader.dataset)
        
    
