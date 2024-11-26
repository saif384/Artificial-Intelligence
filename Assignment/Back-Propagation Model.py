import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    return output * (1 - output)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        self.W_input_hidden = np.random.uniform(-1, 1, (input_dim, hidden_dim))
        self.b_hidden = np.random.uniform(-1, 1, hidden_dim)
        self.W_hidden_output = np.random.uniform(-1, 1, (hidden_dim, output_dim))
        self.b_output = np.random.uniform(-1, 1, output_dim)
        self.lr = lr

    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.W_input_hidden) + self.b_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.W_hidden_output) + self.b_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backpropagate(self, X, y):
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.W_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.W_hidden_output += self.lr * np.dot(self.hidden_output.T, output_delta)
        self.b_output += self.lr * output_delta.sum(axis=0)

        self.W_input_hidden += self.lr * np.dot(X.T, hidden_delta)
        self.b_hidden += self.lr * hidden_delta.sum(axis=0)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i, (inputs, target) in enumerate(zip(X, y)):
                inputs = inputs.reshape(1, -1)
                target = target.reshape(1, -1)
                self.forward_pass(inputs)
                self.backpropagate(inputs, target)

    def predict(self, X):
        return self.forward_pass(X)

def get_training_data():
    print("Select a logic gate to train:\n1. AND\n2. OR\n3. XOR")
    gate = input("Enter your choice: ").strip().upper()

    if gate == 'AND':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
    elif gate == 'OR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
    elif gate == 'XOR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
    else:
        raise ValueError("Invalid gate selection.")

    return X, y, gate

if __name__ == "__main__":
    print("Backpropagation Training Algorithm")

    input_dim = 2
    hidden_dim = int(input("Number of hidden layer neurons (e.g., 2 for XOR): "))
    output_dim = 1
    lr = float(input("Learning rate (e.g., 0.1): "))
    epochs = int(input("Enter the number of epochs: "))

    X, y, gate = get_training_data()

    nn = NeuralNetwork(input_dim, hidden_dim, output_dim, lr)
    nn.train(X, y, epochs)

    print(f"Neural network trained successfully for {gate} gate.")

    while True:
        test_input = list(map(int, input("Enter inputs to test (e.g., 1 0): ").split()))
        prediction = nn.predict(np.array(test_input).reshape(1, -1))
        print(f"Predicted output: {np.round(prediction)[0][0]}")

        cont = input("Test another input? (y/n): ").strip().lower()
        if cont == 'n':
            break
