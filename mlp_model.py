import numpy as np
from numpy import ndarray
from typing import Literal

class MLP:

    class Layer:
        random_init_limits = (-0.20, +0.20)

        def __init__(self, n_inputs: int, n_outputs: int):
            lower_bound, upper_bound = self.random_init_limits
            self.w = np.random.uniform(lower_bound, upper_bound, (n_inputs, n_outputs))
            self.b = np.random.uniform(lower_bound, upper_bound, n_outputs)

            self.z = np.array([[]])
            self.g_w_avg = np.zeros((n_inputs, n_outputs))  # RMSProp moving average for weights
            self.g_b_avg = np.zeros(n_outputs)             # RMSProp moving average for biases
    
    def __init__(self, layers_sizes: list[int], activation : Literal["tanh", "softmax", "sigmoid", "relu"] = "tanh" ):
        layers: list[MLP.Layer] = []

        for i in range(len(layers_sizes) - 1):
            mlp_layer_connection = MLP.Layer(layers_sizes[i], layers_sizes[i + 1])
            layers.append(mlp_layer_connection)

        self.layers = layers

        if activation == "tanh":
            self.act_fn = lambda x: np.tanh(x)
            self.d_act_fn = lambda x: (1 + x) * (1 - x)
        elif activation == "softmax":
            self.act_fn = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
            self.d_act_fn = lambda x: x * (1 - x)
        elif activation == "sigmoid":
            self.act_fn = lambda x: 1 / (1 + np.exp(-x))
            self.d_act_fn = lambda x: self.act_fn(x) * (1 - self.act_fn(x)) 
        elif activation == "relu":
            self.act_fn = lambda x: np.maximum(0, x)
            self.d_act_fn = lambda x: np.where(x > 0, 1, 0)

    def print_parameters(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer {i + 1}')
            print(f'Weights: {layer.w}')
            print(f'Biases: {layer.b}')
            print()
    
    def forward(self, x: ndarray) -> ndarray:
        layers = self.layers
        z = x
        for layer in layers:
            zin = np.dot(z, layer.w) + layer.b
            z = self.act_fn(zin)
            layer.z = z
            
        return z
    
    def backward(self, x: ndarray, y: ndarray, output: ndarray, alpha, beta=0.9) -> None:
        g = output - y

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            g_w = np.dot(prev_layer.z.T, g)
            g_b = np.sum(g, axis=0)

            # Update RMSProp moving averages
            layer.g_w_avg = beta * layer.g_w_avg + (1 - beta) * g_w ** 2
            layer.g_b_avg = beta * layer.g_b_avg + (1 - beta) * g_b ** 2

            # RMSProp weight and bias updates
            layer.w -= (alpha * g_w) / (np.sqrt(layer.g_w_avg) + 1e-8)
            layer.b -= (alpha * g_b) / (np.sqrt(layer.g_b_avg) + 1e-8)

            # Backpropagate to previous layer
            g = np.dot(g, layer.w.T) * self.d_act_fn(prev_layer.z)

        input_layer = self.layers[0]
        g_w = np.dot(x.T, g)
        g_b = np.sum(g, axis=0)

        input_layer.g_w_avg = beta * input_layer.g_w_avg + (1 - beta) * g_w ** 2
        input_layer.g_b_avg = beta * input_layer.g_b_avg + (1 - beta) * g_b ** 2

        input_layer.w -= (alpha * g_w) / (np.sqrt(input_layer.g_w_avg) + 1e-8)
        input_layer.b -= (alpha * g_b) / (np.sqrt(input_layer.g_b_avg) + 1e-8)  

    def train(self, x: ndarray, y: ndarray, learning_rate = 0.01, tolerated_error = 1e-8, max_epochs = 10000):
        epoch = 0
        error = float('inf')
        while error > tolerated_error and epoch < max_epochs:
            epoch += 1
            output = self.forward(x)
            self.backward(x, y, output, learning_rate)
            error = 0.5 * np.sum((output - y) ** 2)
            if epoch % 500 == 0:
                print(f'Error: {error}')

        return error, epoch                

    def predict(self, x):
        return self.forward(x)