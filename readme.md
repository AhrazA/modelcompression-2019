## Thesis Test Bench

### Prerequisites

* Python 3.6+
* Pip3
* Virtualenv (optional)

### Setup

To get started, set up a virtualenv (if so desired):
  
    ~ virtualenv venv
    ~ . venv/bin/activate

Next, install all dependencies:

    ~ pip install -r requirements.txt

Everything should now be ready.

### Usage

#### Train a model from scratch

In order to re-train one of the available models from scratch (without pruning or quantization), simply run the equivalent `*_classifier.py` file. Example:

    ~ python cifar_classifier.py --save-model

By default, this will train a model and save it to `models/cifar_classifier.pt`.

#### Evaluate Pruning

##### Step 1
In order to prune and evaluate a pruned model, ensure that a trained model exists. If you are not sure how to do this, follow the instructions [here](#train-a-model-from-scratch).

##### Step 2
Once confirming a trained model exists, ensure that there exists an appropriate test time configuration for that model. This can be found in `main.py`, under the `configurations` variable. *This will change in the future to a more robust solution.*

##### Step 3 (WILL CHANGE)
Run the command:

    python main.py

You will be presented with a list of available configurations:

    Select a model type to prune. Models available:
        0:{'model': <class 'cifar_classifier.MaskedCifar'>, 'dataset': <class 'torchvision.datasets.cifar.CIFAR10'>, 'transforms': [ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))], 'loss_fn': <function cross_entropy at 0x000001C3C9EB1E18>}
        
        1:{'model': <class 'mnist_classifier.MaskedMNist'>, 'dataset': <class 'torchvision.datasets.mnist.MNIST'>, 'transforms': [ToTensor(), Normalize(mean=(0.1307,), std=(0.3081,))], 'loss_fn': <function nll_loss at 0x000001C3C9EB1AE8>}

Select the appropraite configuration by selecting the correct index:

    Enter selected index and press enter: 0

Then, select the appropriate trained model parameters to load:

    Select a model params file to load by index. Models available:
        0:cifar_classifier.pt
        1:mnist_classifier.pt
        2:mnist_cnn.pt
    
    Enter selected index and press enter: 0

Finally, input the desired pruning percentage:

    Select pruning percentage: (0-100)%: 50

The script will evaluate the pre-pruned model, prune the model, print relevant pruning statistics by layer, and finally evaluate the pruned model.

    Loading file cifar_classifier.pt for model {'model': <class 'cifar_classifier.MaskedCifar'>, 'dataset': <class 'torchvision.datasets.cifar.CIFAR10'>, 'transforms': [ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))], 'loss_fn': <function cross_entropy at 0x000001C3C9EB1E18>}

    Testing pre-pruned model..

    Test set: Average loss: 1.1923, Accuracy: 5865/10000 (59%)

    Pruning model..
    Layer 1 | Conv layer | 15.11% parameters pruned
    Layer 2 | Conv layer | 26.54% parameters pruned
    Layer 3 | Linear layer | 56.20% parameters pruned
    Layer 4 | Linear layer | 30.25% parameters pruned
    Layer 5 | Linear layer | 18.21% parameters pruned
    Final pruning rate: 49.81%

    Evaluating pruned model..

    Test set: Average loss: 1.2210, Accuracy: 5714/10000 (57%)

