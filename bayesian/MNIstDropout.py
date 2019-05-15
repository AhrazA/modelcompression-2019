from __future__ import print_function

import sys
sys.path.append('/mnt/home/a318599/Bayesnn/masters-thesis-2019')

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from classifier import Classifier
from classifier_utils import setup_default_args
from bayesian.ConcreteDropoutLinear import ConcreteDropoutLinear, ConcreteDropoutConvolutional
from bayesian.utils import heteroscedastic_loss
from mnist_classifier import MaskedMNist
from cifar_classifier import MaskedCifar
from classifier import Classifier

class MaskedConcreteCifar(MaskedCifar):
    def __init__(self, weight_regularizer, dropout_regularizer, **kwargs):
        super(MaskedConcreteCifar, self).__init__()

class MaskedConcreteMNist(MaskedMNist):
    def __init__(self, weight_regularizer, dropout_regularizer, **kwargs):
        super(MaskedConcreteMNist, self).__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self.conv_drop1 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conv_drop2 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.conc_drop1 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.conc_drop_mu = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.linear_mu = nn.Linear(10, 1)
        self.linear_logvar = nn.Linear(10, 1)

    def forward(self, x):
        regularization = torch.empty(6, device=x.device)

        x, regularization[0] = self.conv_drop1(x, nn.Sequential(self.conv1, self.relu))
        x = F.max_pool2d(x, 2, 2)

        x, regularization[1] = self.conv_drop2(x, nn.Sequential(self.conv2, self.relu))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)

        x, regularization[2] = self.conc_drop1(x, nn.Sequential(self.fc1, self.relu))
        x, regularization[3] = self.conc_drop2(x, nn.Sequential(self.fc2, self.softmax))

        mean, regularization[4] = self.conc_drop_mu(x, self.linear_mu)
        log_var, regularization[5] = self.conc_drop_logvar(x, self.linear_logvar)

        return mean, log_var, regularization.sum()

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def test(Y_val, means, logvar, K_test=20):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D
    """
    Y_val = Y_val.numpy()
    k = K_test
    N = Y_val.shape[0]
    import pdb
    pdb.set_trace()
    mean = means 
    logvar = logvar
    test_ll = -0.5 * np.exp(-logvar) * (mean - Y_val.squeeze())**2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi) #Y_true[None]
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N  # per point predictive probability
    rmse = np.mean((np.mean(mean, 0) - Y_val.squeeze())**2.)**0.5
    return pppp, rmse

def fit_model(model, nb_epoch, dataloader):
    optimizer = optim.Adam(model.parameters())

    for i in range(nb_epoch):
        print("Starting epoch: ", i)
        for idx, (X, Y) in enumerate(dataloader):
            X = X.type('torch.cuda.FloatTensor')
            Y = Y.type('torch.cuda.FloatTensor')
            N = X.shape[0]
            wr = l**2. / N
            dr = 2. / N
                
            x = Variable(X).cuda(2)
            y = Variable(Y).cuda(2)
            
            mean, log_var, regularization = model(x)
                        
            loss = heteroscedastic_loss(y, mean, log_var) + regularization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 300 == 0: print("Loss: ", loss.item())            
            
    return model

def run(model, train_dataloader, test_dataloader, K_test=20):
    results = []
    nb_reps = 3

    model.eval()

    test_iter = iter(test_dataloader)
    
    Y_val = torch.tensor([model(next(test_iter)[0].to('cuda:2')) for j in range(K_test)])
    MC_samples = Y_val

    means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, len(test_iter)).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, len(test_iter)).cpu().data.numpy()

    pppp, rmse = test(Y_val, means, logvar, K_test)
    epistemic_uncertainty = np.var(means, 0).mean(0)
    logvar = np.mean(logvar, 0)
    aleatoric_uncertainty = np.exp(logvar).mean(0)
    ps = np.array([torch.sigmoid(module.p_logit).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p_logit')])
    return (rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)

if __name__ == '__main__':
    device = 'cuda:2'

    train_data = test_data = datasets.MNIST(
        './data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
    )

    test_data = datasets.MNIST(
        './data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    pre_trained_weights = torch.load('./models/mnist_classifier.pt', map_location=torch.device(device))
    nb_epoch = 20
    nb_reps = 5
    rep_results = []

    for i in range(nb_reps):
        l = 1e-4
        N = len(train_loader)
        wr = l**2. / N
        dr = 2. / N

        model = MaskedConcreteMNist(wr, dr).to(device)
        
        state_dict = model.state_dict()
        state_dict.update(pre_trained_weights)
        model.load_state_dict(state_dict)

        # fit_model(model, nb_epoch, train_loader)
        rep_results.append(run(model, train_loader, test_loader))
        print(rep_results)
    
    test_mean = np.mean([r[0] for r in rep_results])
    test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
    ps = np.mean([r[1] for r in rep_results], 0)
    aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
    epistemic_uncertainty = np.mean([r[3] for r in rep_results])

    print("Numb epoch", '-', 'test mean', 'test std err', 'ps', '-', 'a uncertainty', 'e uncertainty')
    print(nb_epoch, '-', test_mean, test_std_err, ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)