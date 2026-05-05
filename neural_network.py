"""
neural_network.py
Track 1 — Bare Metal AI Predictor
A simple Multi-Layer Perceptron built from scratch using NumPy.
"""

import numpy as np


class NeuralNetwork:
    """A simple Multi-Layer Perceptron built from scratch using NumPy."""

    def __init__(self):
        # Seed for reproducible results
        np.random.seed(1)
        # Weights for a 2-input, 1-output network; shape (2, 1)
        self.weights = 2 * np.random.random((2, 1)) - 1

    def sigmoid(self, x):
        """Activation function: squashes any value into (0, 1)."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function used in backpropagation."""
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, iterations):
        """Train the model by adjusting weights based on error.

        Args:
            training_inputs:  numpy array of shape (n_samples, 2)
            training_outputs: numpy array of shape (n_samples, 1)
            iterations:       number of training cycles (int)
        """
        for _ in range(iterations):
            output = self.predict(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustments

    def predict(self, inputs):
        """Pass inputs through the network and return a prediction.

        Args:
            inputs: numpy array of shape (n_samples, 2) or (2,)

        Returns:
            numpy array of sigmoid outputs in the range (0, 1)
        """
        inputs = np.array(inputs, dtype=float)
        return self.sigmoid(np.dot(inputs, self.weights))
