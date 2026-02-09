import sys
sys.path.append('.')
from NN import NeuralNetwork, categorical_cross_entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train_encoded.csv")
x = train.drop('price_class', axis=1).values.T
y = pd.get_dummies(train['price_class']).values.T

BATCH_SIZE = 64

nn_relu = NeuralNetwork(input_size=x.shape[0], hidden_size=32, output_size=y.shape[0], activation_type='relu')
nn_sigmoid = NeuralNetwork(input_size=x.shape[0], hidden_size=32, output_size=y.shape[0], activation_type='sigmoid')

# CHANGED: Lists to store metrics for plotting
relu_loss_history = []
sigmoid_loss_history = []
relu_acc_history = []
sigmoid_acc_history = []

for epoch in range(200):
    running_relu_loss = 0
    running_sigmoid_loss = 0
    running_acc_relu = 0
    running_acc_sigmoid = 0

    for i in range(0, x.shape[1], BATCH_SIZE):
        x_batch = x[:, i:i+BATCH_SIZE]
        y_batch = y[:, i:i+BATCH_SIZE]

        pred_relu = nn_relu.forward(x_batch)
        pred_sigmoid = nn_sigmoid.forward(x_batch)

        loss_relu = categorical_cross_entropy(y_batch, pred_relu)
        loss_sigmoid = categorical_cross_entropy(y_batch, pred_sigmoid)

        running_relu_loss += loss_relu
        running_sigmoid_loss += loss_sigmoid

        acc_relu = np.mean(np.argmax(pred_relu, axis=0) == np.argmax(y_batch, axis=0))
        acc_sigmoid = np.mean(np.argmax(pred_sigmoid, axis=0) == np.argmax(y_batch, axis=0))
        running_acc_relu += acc_relu
        running_acc_sigmoid += acc_sigmoid

        nn_relu.backward(y_batch, pred_relu)
        nn_sigmoid.backward(y_batch, pred_sigmoid) 
        nn_relu.step(lr=0.01)
        nn_sigmoid.step(lr=0.01)

    total_steps = (x.shape[1] / BATCH_SIZE)
    epoch_relu_acc = running_acc_relu / total_steps 
    epoch_sigmoid_acc = running_acc_sigmoid / total_steps
    epoch_relu_loss = running_relu_loss / total_steps
    epoch_sigmoid_loss = running_sigmoid_loss / total_steps
    
    relu_loss_history.append(epoch_relu_loss)
    sigmoid_loss_history.append(epoch_sigmoid_loss)
    relu_acc_history.append(epoch_relu_acc)
    sigmoid_acc_history.append(epoch_sigmoid_acc)
    
    print(f"Epoch {epoch}, ReLU Loss: {epoch_relu_loss:.5f}, ReLU Acc: {epoch_relu_acc:.4f}, Sigmoid Loss: {epoch_sigmoid_loss:.5f}, Sigmoid Acc: {epoch_sigmoid_acc:.4f}")

    print(f"Relu Grad History Layer 1: {nn_relu.grad_history[0][-1] if nn_relu.grad_history[0] else 0}, Layer 2: {nn_relu.grad_history[1][-1] if nn_relu.grad_history[1] else 0}")
    print(f"Sigmoid Grad History Layer 1: {nn_sigmoid.grad_history[0][-1] if nn_sigmoid.grad_history[0] else 0}, Layer 2: {nn_sigmoid.grad_history[1][-1] if nn_sigmoid.grad_history[1] else 0}")

test = pd.read_csv("test_encoded.csv")
x_test = test.drop('price_class', axis=1).values.T
y_test = pd.get_dummies(test['price_class']).values.T
pred_test_relu = nn_relu.forward(x_test)
pred_test_sigmoid = nn_sigmoid.forward(x_test)
test_acc_relu = np.mean(np.argmax(pred_test_relu, axis=0) == np.argmax(y_test, axis=0))
test_acc_sigmoid = np.mean(np.argmax(pred_test_sigmoid, axis=0) == np.argmax(y_test, axis=0))
print(f"Test Accuracy ReLU: {test_acc_relu:.4f}, Sigmoid: {test_acc_sigmoid:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(relu_loss_history, label='ReLU Loss')
plt.plot(sigmoid_loss_history, label='Sigmoid Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(relu_acc_history, label='ReLU Accuracy')
plt.plot(sigmoid_acc_history, label='Sigmoid Accuracy')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()