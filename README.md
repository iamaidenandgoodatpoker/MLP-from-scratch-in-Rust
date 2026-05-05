# MLP-from-scratch-in-Rust

Aiden Young: aideny2


Implement Fully connected multilayer perceptron in Rust with only ndarray library.

Weight and bias initialization, forward pass, back propagation, gradient descent, training/testing logic we all be handled manually.
Our goal is to train the network on the MNIST dataset which consists of 70,000 28x28 grayscale handwritten digits. We aim to hit at least 95% accuracy on testing.

I wanted to do this because I like data science and have implemented a neural network from scratch (with only numpy) in Python. It had good accuracy but it was slow. Rust is a lot more. efficient. Another goal I have is for training to be fast. In order to achieve this, the plan to reduce cloning values as much as possible and to vectorize data instead of looping.

Used He-initialization, ReLU activation function, Soft-max on the final layer, cross entropy loss function, and mini batch gradient descent.


-Initialize Neural Network
  Neural network consists of Weights and Biases.

  Use random seed to keep initialized weights the same between trials.
  
  Weights are initialized with dimensions [output, input] with random values from the standard normal distribution.
  Every value is then scaled by He-initialization which is sqrt(2/n).

  Biases are initialized to 0 and have dimension [output,]


-Forward Pass (Planned by first checkpoint, NOT COMPLETED) 
  My  data (28x28 = 784 pixels) will be passed through the Neural Network.

  Save all activations (including input data) to a vector which will be returned.
  We will use this to calculate gradients.

  Each activation matrix will roughly look like this: (input @ weight + bias).ReLU.
  On the last layer we will use SoftMax instead of ReLU. This function is good for probabilistic predicts because all values sum to 1.


-Calculate Gradients (Planned by second checkpointm NOT COMPLETED)
  We need to train the network by finding the error of each node the "blame" each.
  We run the forward pass with our given data and that is the model's prediction.
  
  Our loss function is Cross Entropy Loss and this combined with SoftMax in the final layer gives a simple error.
  In the final layer the error is A - y where y is a 10 element array and has a 1 in the index of the correct digit and A is the activation of the final layer.

  Calculate dw (delta weight) with A_prev (previous layer's activation) * dz (error). We get this through the chain rule.
  Calculate db (delta biaas) with dz. We also get this through the chain_rule.

  We need to calculate error now which is the formula dz_prev = dz * w * f'(z). This is again the result of the chain rule. dz_prev is the error of the previous layer.

  Propagate this all backwards until we reach the input layer.

  With our gradients, we scale it by the learning rate and subtract it from our weights and biases. The hope is to find a local minimum for our loss function.


-Training/Testing
  We have 60,000 images to train with and 10,000 to test with
  We loop through 60,000 images in batches to calculate gradients. This keep training more stable so it doesn't overfit for a single image per loop.

  Loop over entire 60,000 for any amount of epochs.

  Test the model after each epoch by plugging in the 10,000 images that the network has never seen before and record the accuracy.


Challenges:

Never used ndarray before. Will need a lot of time to learn about its functions.

We need to vectorize everything. Instead using a for loop for everything, we want to use big matrices. This is a lot harder than simple loops because what is happening is not as obivous.
Since we are programming this in Rust for the sake of performance speed, we need to optimize as much as possible.

We have to pay careful attention to ownership rules and borrowing because we do not want to waste computation copying what could be hundreds of thousands of weights or images.

We want to make the network flexible. It will not have a set amount of hidden layers or nodes so rather than individually defining each layer, we have to loop through values.
This is a lot harder because we don't see exactly what is happening as we move through the network and have to generalize everything. This is a great exercise though.

Anything else to keep things running efficiently while maintaining clean code.

My teammate knows nothing about neural networks so I'm not sure what to do about that.


Inspiration:

I implemented a neural network from scratch in Python last semester. The main problem with it was that it was slow and the code was very ugly.
In addition, it only supported fixed stuctures (which was easier to code) so the hyperparameters were a lot harder to adjust.


FINAL RESULTS:

Acheived 98.11% accuracy.

Trained a lot faster than a Python implementation with numpy

Epoch:  1 | Test Accuracy: 0.9242%
Epoch:  2 | Test Accuracy: 0.9320%
Epoch:  3 | Test Accuracy: 0.9417%
...
Epoch: 88 | Test Accuracy: 0.9792%
Epoch: 89 | Test Accuracy: 0.9806%
Epoch: 90 | Test Accuracy: 0.9811%
