from __future__ import print_function

# import argparse

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
from mnist_classifier import MaskedMNist
from cifar_classifier import MaskedCifar
from classifier import Classifier
from torch.utils.tensorboard import SummaryWriter
import configurations

def heteroscedastic_loss(mean, log_var, true):
    precision = torch.exp(-log_var)
    true = true.type('torch.cuda.FloatTensor')
    sum_ = torch.sum(precision * (true - mean)**2 + log_var, 1)
    out = torch.mean(sum_, 0)
    return out

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def run(model, train_dataloader, test_dataloader, K_test=20):
    results = []
    nb_reps = 3

    model.eval()

    test_iter = iter(test_dataloader)
    
    Y_val = torch.tensor([model(next(test_iter)[0].to('cuda:0')) for j in range(K_test)])
    MC_samples = Y_val

    means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, len(test_iter)).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, len(test_iter)).cpu().data.numpy()

    pppp, rmse = test(Y_val, means, logvar, K_test)
    epistemic_uncertainty = np.var(means, 0).mean(0)
    logvar = np.mean(logvar, 0)
    aleatoric_uncertainty = np.exp(logvar).mean(0)
    ps = np.array([torch.sigmoid(module.p_logit).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p_logit')])
    return (rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)

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
                
            x = Variable(X).cuda(0)
            y = Variable(Y).cuda(0)
            
            mean, log_var, regularization = model(x)
                        
            loss = heteroscedastic_loss(y, mean, log_var) + regularization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 300 == 0: print("Loss: ", loss.item())            
            
    return model

    # nb_epoch = 20
    # nb_reps = 5
    # rep_results = []

    # for i in range(nb_reps):
    #     l = 1e-4
    #     N = len(train_loader)
    #     wr = l**2. / N
    #     dr = 2. / N

        

    #     fit_model(model, nb_epoch, train_loader)
    #     rep_results.append(run(model, train_loader, test_loader))
    #     print(rep_results)
    
    # test_mean = np.mean([r[0] for r in rep_results])
    # test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
    # ps = np.mean([r[1] for r in rep_results], 0)
    # aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
    # epistemic_uncertainty = np.mean([r[3] for r in rep_results])

    # print("Numb epoch", '-', 'test mean', 'test std err', 'ps', '-', 'a uncertainty', 'e uncertainty')
    # print(nb_epoch, '-', test_mean, test_std_err, ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)