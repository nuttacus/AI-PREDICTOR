"""
main.py
Track 1 — Bare Metal AI Predictor
Entry point: trains the NeuralNetwork on XOR data and prints predictions.
Run with: python main.py
"""

import numpy as np
from neural_network import NeuralNetwork


def main():
    # --- Training data: XOR logic gate ---
    # Inputs:  [0,0] [0,1] [1,0] [1,1]
    # Outputs:   0     1     1     0
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    training_outputs = np.array([[0], [1], [1], [0]])

    # --- Initialise and train the network ---
    nn = NeuralNetwork()
    print("Training neural network on XOR data (10,000 iterations)...")
    nn.train(training_inputs, training_outputs, iterations=10_000)
    print("Training complete.\n")

    # --- Display predictions for all XOR inputs ---
    print(f"{'Input 1':<10} {'Input 2':<10} {'Raw output':<14} {'Rounded'}")
    print("-" * 44)
    for inputs, expected in zip(training_inputs, training_outputs):
        prediction = nn.predict(inputs)
        raw = float(prediction)
        rounded = round(raw)
        print(f"{inputs[0]:<10} {inputs[1]:<10} {raw:<14.4f} {rounded}  (expected {expected[0]})")

    # --- Interactive single prediction ---
    print("\n--- Custom prediction ---")
    try:
        val1 = float(input("Enter Input 1 (0 or 1): "))
        val2 = float(input("Enter Input 2 (0 or 1): "))
        result = nn.predict(np.array([val1, val2]))
        raw = float(result)
        print(f"\nPrediction: {raw:.4f}  →  rounds to {round(raw)}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")


if __name__ == "__main__":
    main()
