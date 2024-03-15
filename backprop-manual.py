#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:17:00 2024

@author: vishwa
"""

import numpy as np
from numpy.random import default_rng
import tensorflow as tf
import matplotlib.pyplot as plt



# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.imshow(x_train[0],cmap='gray');

# Reshape and normalize the dataset
#x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
#x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

def sigmoid(x):
  # Numerically stable sigmoid function based on
  # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
  
  x = np.clip(x, -500, 500) # We get an overflow warning without this
  
  return np.where(
    x >= 0,
    1 / (1 + np.exp(-x)),
    np.exp(x) / (1 + np.exp(x))
  )


def dsigmoid(x): # Derivative of sigmoid
  return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
  # Numerically stable softmax based on (same source as sigmoid)
  # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
  b = x.max()
  y = np.exp(x - b)
  return y / y.sum()


def cross_entropy_loss(y, yHat):
  return -np.sum(y * np.log(yHat))


def integer_to_one_hot(x, max):
  # x: integer to convert to one hot encoding
  # max: the size of the one hot encoded array
  result = np.zeros(10)
  result[x] = 1
  return result

rng = default_rng(80085)

# Layer sizes
input_size = 784  # 28x28 images
hidden_size = 32
output_size = 10

# Weight and bias initialization
weights = {
    'h1': rng.normal(0, 1/np.sqrt(input_size), (input_size, hidden_size)),
    'h2': rng.normal(0, 1/np.sqrt(hidden_size), (hidden_size, hidden_size)),
    'out': rng.normal(0, 1/np.sqrt(hidden_size), (hidden_size, output_size))
}

biases = {
    'h1': np.zeros(hidden_size),
    'h2': np.zeros(hidden_size),
    'out': np.zeros(output_size)
}



def feed_forward_sample(sample, weights, biases):
    """ Forward pass for a single sample """
    activations = {}

    # Input layer to hidden layer 1
    z_h1 = np.dot(sample, weights['h1']) + biases['h1']
    a_h1 = sigmoid(z_h1)
    activations['h1'] = a_h1

    # Hidden layer 1 to hidden layer 2
    z_h2 = np.dot(a_h1, weights['h2']) + biases['h2']
    a_h2 = sigmoid(z_h2)
    activations['h2'] = a_h2

    # Hidden layer 2 to output layer
    z_out = np.dot(a_h2, weights['out']) + biases['out']
    a_out = softmax(z_out)
    activations['out'] = a_out

    return activations

def feed_forward_dataset(x, y, weights, biases):
    """ Forward pass for the entire dataset """
    losses = []
    y_hats = []

    for sample, label in zip(x, y):
        # Flatten the image and feed forward
        sample = sample.flatten() / 255.0  # Normalizing the input
        activations = feed_forward_sample(sample, weights, biases)
        
        # Convert label to one-hot encoding
        label_one_hot = integer_to_one_hot(label, output_size)
        
        # Compute the loss
        loss = cross_entropy_loss(label_one_hot, activations['out'])
        losses.append(loss)
        
        # The prediction is the index of the max softmax output
        y_hat = np.argmax(activations['out'])
        y_hats.append(y_hat)

    # Calculate the accuracy
    accuracy = np.mean(np.array(y_hats) == y)
    average_loss = np.mean(losses)

    return average_loss, accuracy


def backpropagation(sample, label_one_hot, activations, weights):
    """ Compute the gradient of the loss function with respect to each weight and bias. """
    weight_gradients = {}
    bias_gradients = {}
    
    # Output layer error δ^(L)
    error_out = (activations['out'] - label_one_hot)
    
    # Backpropagate to hidden layer 2
    error_h2 = np.dot(error_out, weights['out'].T) * dsigmoid(activations['h2'])
    
    # Backpropagate to hidden layer 1
    error_h1 = np.dot(error_h2, weights['h2'].T) * dsigmoid(activations['h1'])
    
    # Gradient for output layer weights
    weight_gradients['out'] = np.outer(activations['h2'], error_out)
    
    # Gradient for hidden layer 2 weights
    weight_gradients['h2'] = np.outer(activations['h1'], error_h2)
    
    # Gradient for hidden layer 1 weights
    weight_gradients['h1'] = np.outer(sample, error_h1)
    
    # Bias gradients
    bias_gradients['out'] = error_out
    bias_gradients['h2'] = error_h2
    bias_gradients['h1'] = error_h1
    
    return weight_gradients, bias_gradients


def train_one_sample(sample, label, weights, biases, learning_rate=0.003):
    sample = sample.flatten() / 255.0  # Normalize the sample
    label_one_hot = integer_to_one_hot(label, output_size)
    
    # Forward pass
    activations = feed_forward_sample(sample, weights, biases)
    
    # Backward pass
    weight_gradients, bias_gradients = backpropagation(sample, label_one_hot, activations, weights)
    
    # Update weights and biases
    for layer in weights:
        weights[layer] -= learning_rate * weight_gradients[layer]
        biases[layer] -= learning_rate * bias_gradients[layer].flatten()


def train_one_epoch(x_train, y_train, weights, biases, learning_rate=0.003):
    print("Training for one epoch over the training dataset...")
    
    for sample, label in zip(x_train, y_train):
        train_one_sample(sample, label, weights, biases, learning_rate)
    
    print("Finished training.\n")


def test_and_train(x_train, y_train, x_test, y_test, weights, biases):
    for i in range(10):
        # Train for one epoch
        train_one_epoch(x_train, y_train, weights, biases)
        
        # Test the accuracy on the test dataset
        test_loss, test_accuracy = feed_forward_dataset(x_test, y_test, weights, biases)
        print(f"Epoch {i+1}: Test Loss = {test_loss:.2f}, Test Accuracy = {test_accuracy:.2%}\n")


test_and_train(x_train, y_train, x_test, y_test, weights, biases)

