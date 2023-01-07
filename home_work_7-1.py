# HW# 7  Parity-3 problem NN_520 Course 2022

#import required libraries
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt

#define activation functions and their derivatives
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


#building the model
def build_model(x, output_dim=1):
    model = {}
    
    model["w1"] = np.random.randn(x.shape[1], output_dim)
    model["b1"] = np.random.randn(output_dim, 1)

    return model 

#Building the layers of the model
def forward(x, model):
    # output layer 
    w1 = model["w1"]
    b1 = model["b1"]
    z1 = x.dot(w1) + b1
    out = sigmoid(z1) 
        
    pred = [0.0 if (i[0] < 0.5) else 1.0 for  i in out]
    return z1, np.array(pred) 

#train the model
def train(model, x, y):
    lr = 0.01
    total_iter = 50000
    error_grad = 0
    losses = []
    iters = []
    accs = []

    for iter in range(total_iter):
        z1, pred = forward(x, model)
        error_grad = np.expand_dims((pred - y)/8, axis=1) 

        delta_1 = error_grad * sigmoid_deriv(z1)
        dw1 = np.dot(x.T, delta_1)
        db1 = np.sum(delta_1, axis=0)

        model["w1"] -= lr * dw1
        model["b1"] -= lr * db1

        if iter % 1000 == 0:
            loss = calculate_loss(y, pred)
            accs.append(check_accuracy(y, pred))
            losses.append(loss)
            iters.append(iter)
            print("Loss after %d iteration %f" % (iter, loss))
        
    return model, losses, iters, accs 


def main(): 
    x = np.array([
        #x1x2, x1x3, x2x3, x1x2x3
        [1, 1, 1, 1, 1, 1, 1], 
        [1, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0], 

        [0, 0, 0, 0, 0, 0, 0], 
        [1, 0, 1, 0, 1, 0, 0], 
        [0, 1, 1, 0, 0, 1, 0], 
        [1, 1, 0, 1, 0, 0, 0]])

    y = [1, 1, 1, 1, 0, 0, 0, 0]

    model = build_model(x)
    model, losses, iters, accs = train(model, x, y)

    print(model["w1"])
    print(model["b1"])
    plt.plot(iters, losses, color='blue', marker='o', markersize=5)
    plt.title('Learning Curve')
    plt.xlabel('Training iteration')
    plt.ylabel('Losses')
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