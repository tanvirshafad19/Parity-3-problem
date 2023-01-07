# HW# 7  Parity-3 problem NN_520 Course 2022
# Network with hidden layer
# figure c in the problem statement


import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt 

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) 

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (np.exp(x)-np.exp(-x)) /(np.exp(x) + np.exp(-x))

def tanh_deriv(x):
    return 1 - (tanh(x))**2

def calculate_loss(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)

def check_accuracy(y_true, y_pred):

    return sum(y_true[i] == y_pred[i] for i in range(0, 8)) * (100/8)


def build_model(x, hidden_dim=3, output_dim=1):
    model = {}
    
    model["w1"] = np.random.randn(x.shape[1], hidden_dim)
    model["b1"] = np.random.randn(1, hidden_dim)

    model["w2"] = np.random.randn(hidden_dim, output_dim)
    model["b2"] = np.random.randn(output_dim, 1)

    return model 


def forward(x, model):
    # 2nd layer 
    w1 = model["w1"]
    b1 = model["b1"]
    z1 = x.dot(w1) + b1
    a1 = sigmoid(z1) 

    # 3rd layer 
    w2 = model["w2"]
    b2 = model["b2"]
    z2 = a1.dot(w2) + b2
    out = sigmoid(z2)
        
    pred = [0.0 if (i[0] < 0.5) else 1.0 for  i in out]
    return z1, a1, z2, np.array(pred) 


def train(model, x, y):
    lr = 0.03
    total_iter = 400000
    error_grad = 0
    losses = []
    iters = []
    accs = []

    for iter in range(total_iter):
        z1, a1, z2, pred = forward(x, model)
        error_grad = np.expand_dims((pred - y)/8, axis=1) 

        delta_2 = error_grad * sigmoid_deriv(z2)
        dw2 = np.dot(a1.T, delta_2)
        db2 = np.sum(delta_2, axis=0)

        delta_1 = np.dot(delta_2, model["w2"].T) * sigmoid_deriv(a1)
        dw1 = np.dot(x.T, delta_1)
        db1 = np.sum(delta_1, axis=0)

        model["w1"] -= lr * dw1
        model["b1"] -= lr * db1
        model["w2"] -= lr * dw2
        model["b2"] -= lr * db2 

        if iter % 10000 == 0:
            loss = calculate_loss(y, pred)
            accs.append(check_accuracy(y, pred))
            losses.append(loss)
            iters.append(iter)
            print("Loss after %d iteration %f" % (iter, loss))
        
    return model, losses, iters, accs 


def main(): 
    x = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], 
         [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])

    y = [1, 1, 1, 1, 0, 0, 0, 0]

    model = build_model(x)
    model, losses, iters, accs = train(model, x, y)
    print(model)
    plt.plot(iters,losses, color='blue', marker='o', markersize=5)
    plt.title('Learning Curve')
    plt.xlabel('Training iteration')
    plt.ylabel('losses')
    plt.grid()
    plt.show()
    
    plt.plot(iters,accs, color='blue', marker='o', markersize=5)
    plt.title('Learning Curve')
    plt.xlabel('Training iteration')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()