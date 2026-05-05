# cs128hproject

Implement Fully connected multilayer perceptron in Rust with only ndarray library

Weight and bias initialization, forward pass, back propagation, gradient descent, training/testing logic have all be handled manually.

Used He-initialization, ReLU activation function, Soft-max on the final layer, cross entropy loss function, and mini batch gradient descent.

Acheived 98.11% accuracy.

Trained a lot faster than a Python implementation with numpy

Epoch:  1 | Test Accuracy: 0.9242%
Epoch:  2 | Test Accuracy: 0.9320%
Epoch:  3 | Test Accuracy: 0.9417%
...
Epoch: 88 | Test Accuracy: 0.9792%
Epoch: 89 | Test Accuracy: 0.9806%
Epoch: 90 | Test Accuracy: 0.9811%
