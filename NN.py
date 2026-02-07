import numpy as np

def softmax(x):
    # Subtracting the max for numerical stability (prevents overflow)
    # results in the same prob dist so no harm done!
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)

def sigmoid(x):
    # again done for stability, it all results in a number extremely close to +1 or -1
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    z = sigmoid(x)
    return z * (1 - z)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def categorical_cross_entropy(y_true, y_pred):
    # y_true is expected to be one-hot encoded (classes x Batch)
    epsilon = 1e-11
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


class Layer:
    def __init__(self, in_shape, out_shape):
        # np.sqrt(2.0 / in_shape) is taken from the pytorch documentation and is known as He intialization
        # https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        # You can comment it out however it will lead to worse results
        self.weights = np.random.randn(out_shape, in_shape) * np.sqrt(2.0 / in_shape)
        self.bias = np.zeros((out_shape, 1))
        self.net = None
        self.output = None
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
"""
This design is slightly inspired from pytorch
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_type='relu'):
        self.l1 = Layer(input_size, hidden_size)
        self.l2 = Layer(hidden_size, hidden_size) # maybe change this to whatever your heart desires
        self.l3 = Layer(hidden_size, output_size)
        self.layers = [self.l1, self.l2, self.l3]
        self.activation_type = activation_type
        self.act = sigmoid if activation_type == 'sigmoid' else relu
        self.act_der = sigmoid_derivative if activation_type == 'sigmoid' else relu_derivative
        self.grad_history = {0: [], 1: []}

    def forward(self, x):
        self.input = x
        z1 = self.l1.forward(x)
        a1 = self.act(z1)
        self.l1.net = z1
        self.l1.output = a1

        z2 = self.l2.forward(a1)
        a2 = self.act(z2)
        self.l2.net = z2
        self.l2.output = a2

        z3 = self.l3.forward(a2)
        a3 = softmax(z3)
        self.l3.net = z3
        self.l3.output = a3

        return a3

    def backward(self, target, prediction):
        # This trick is takne from https://benjaminroland.onrender.com/coding&data/the-rational-of-softmax/
        delta = prediction - target
        batch_size = target.shape[1]

        for i in range(len(self.layers)-1, -1, -1): # fancy ahh way of starting from last layer
            # remember if i=0 we are at the input layer otherwise we've got stuff to do
            curr_layer = self.layers[i]
            prev_out = self.layers[i-1].output if i > 0 else self.input

            curr_layer.grad_w = np.dot(delta, prev_out.T) / batch_size
            curr_layer.grad_b = np.mean(delta, axis=1, keepdims=True)

            if i < 2:
                self.grad_history[i].append(np.mean(np.abs(curr_layer.grad_w)))
            # would be kind of interesting to propagate the gradiants to the input
            if i > 0:
                # Delta_prev = (W_curr.T dot Delta_curr) * activation_derivative(z_prev)
                # this is just carrying that gradiant back for us
                delta = np.dot(curr_layer.weights.T, delta) * self.act_der(self.layers[i-1].net)

    def step(self, lr=0.01):
        for layer in self.layers:
            layer.weights -= lr * layer.grad_w
            layer.bias -= lr * layer.grad_b

