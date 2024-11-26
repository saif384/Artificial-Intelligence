import numpy as np


class PerceptronModel:
    def __init__(self, num_inputs, lr=0.1, max_epochs=100):
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = np.zeros(num_inputs)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else 0

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            for i, inputs in enumerate(X):
                net_input = np.dot(inputs, self.weights) + self.bias
                output = self.activation(net_input)
                error = y[i] - output
                # Update rule
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error

    def predict(self, X):
        net_input = np.dot(X, self.weights) + self.bias
        return self.activation(net_input)


def get_data():
    print("Select a logic gate for training:\n1. AND\n2. OR")
    choice = input("Enter your choice: ").strip().upper()

    if choice == 'AND':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
    elif choice == 'OR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])

    else:
        print("Invalid choice! Please select a valid gate.")
        exit()

    return X, y, choice


if __name__ == "__main__":
    print("Perceptron Training Algorithm")
    input_size = 2
    lr = float(input("Enter learning rate (e.g., 0.1): "))
    epochs = int(input("Enter number of epochs: "))

    # Load training data
    X, y, gate = get_data()

    # Initialize and train the perceptron
    perceptron = PerceptronModel(num_inputs=input_size, lr=lr, max_epochs=epochs)
    perceptron.train(X, y)

    print(f"Perceptron has been trained for {gate} gate.")

    # Test phase
    while True:
        test_input = list(map(int, input("Enter inputs to test (e.g., 1 0): ").split()))
        prediction = perceptron.predict(np.array(test_input))
        print(f"Predicted output: {prediction}")

        cont = input("Test another input? (y/n): ").strip().lower()
        if cont == 'n':
            break
