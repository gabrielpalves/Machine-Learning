import scipy.io
import numpy as np
from matplotlib import pyplot


def Theta_transform(layer_size, nn_params):
    """Transform unrolled Theta in a list of Thetas matrices"""
    if isinstance(nn_params, list):
        return nn_params
        
    aux = [-1]
    aux2 = [0]
    Thetas = list()
    for i in range(len(layer_size)-1):
        aux.append((layer_size[i]+1) * (layer_size[i+1]) + sum(aux2))
        aux2.append((layer_size[i]+1) * (layer_size[i+1]))
        aux3 = 1 if i == 0 else 0

        Thetas.append(
            np.reshape(nn_params[aux[i]+aux3:aux[i+1]],
                       (layer_size[i+1], (layer_size[i] + 1)))
        )

    return Thetas


def predict(X, y, layer_size, theta, activation_function, set, printar=False):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network
    """
    
    activations, _ = all_activations(X, layer_size, theta, activation_function)
    prediction = np.argmax(activations[-1], axis=1)
    acc = np.mean(prediction == y) * 100
    if printar:
        print(set + ' Set Accuracy: %f' % acc)
    
    return acc


def debugInitializeWeights(fan_out, fan_in):
    """
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
    connections using a fixed strategy. This will help you later in debugging.
    Returns the initialized weights array given the dimensions.
    """
    
    # Initialize W using "sin"
    # This ensures that W is always of the same values and will be useful for debugging
    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def compute_numerical_gradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences"
    and gives a numerical estimate of the gradient.
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad


def check_NN_gradients(nn_cost_function, activation_function):
    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by the backpropagation code and the numerical gradients
    (computed using compute_numerical_gradient). These two gradient computations should result in
    very similar values.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = np.arange(1, 1+m) % num_labels
    
    # Unroll parameters
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

    # short hand for cost function
    def costFunc(p): return nn_cost_function(p,
                                           num_labels, X, y, activation_function, layer_size=[input_layer_size, hidden_layer_size, num_labels])
    cost, grad = costFunc(nn_params)
    numgrad = compute_numerical_gradient(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in compute_numerical_gradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)



# Activation functions
def sigmoid(z):
    """Activation function: Sigmoid"""
    s = lambda z: 1.0 / (1.0 + np.exp(-z))
    return s(z), s(z) * (1 - s(z))


def tanh(z):
    """Activation function: Hyperbolic Tangent"""
    th = lambda z: (np.exp(2*z) - 1.0) / (np.exp(2*z) + 1.0)
    return th(z), 1.0 - np.power(th(z), 2)


def relu(z):
    """Activation function: Rectified Linear Unit"""
    d = np.copy(z)
    d[z > 0] = 1
    d[z <= 0] = 0
    return np.maximum(np.zeros(z.shape), z), d


def set_activation_function(activation_function):
    match activation_function:
        case 'sigmoid':
            activation_function = sigmoid
        case 'tanh':
            activation_function = tanh
        case 'relu':
            activation_function = relu
        case _:
            activation_function = sigmoid
    return activation_function


def all_activations(X, layer_size, nn_params, activation_function):
    """
    All activations in a list
    Each index i of the list contains the activations of layer i
    """
    Thetas = Theta_transform(layer_size, nn_params)

    activations = list()
    nets = [X]
    for i in range(len(layer_size)):
        if i == 0:
            activations.append(
                np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
            )
        else:
            a = activations[i-1]
            Theta = Thetas[i-1]
            z = a.dot(Theta.T)
            nets.append(z)
            a_i, _ = activation_function(z)
            if i == len(layer_size)-1:
                activations.append(
                    a_i
                )
            else:
                activations.append(
                    np.concatenate([np.ones((a_i.shape[0], 1)), a_i], axis=1)
                )

    return activations, nets


def nn_cost_function(nn_params, num_labels, X, y, activation_function, layer_size):
    """
    Implements the neural network cost function and gradient for a neural 
    network which performs classification. 

    Returns
        cost J for the current weights and
        grad, an "unrolled" vector of the partial derivatives of the concatenation of
    neural network weights Thetas.
    """

    # Reshape nn_params
    Thetas = Theta_transform(layer_size, nn_params)

    m = y.size
                
    activations, nets = all_activations(X, layer_size, Thetas, activation_function)

    # Cálculo do custo J
    y_matrix = np.eye(num_labels)[y]

    h = activations[-1]  # hipótese
    J = (-1 / m) * np.sum((np.log(h) * y_matrix) + np.log(1 - h) * (1 - y_matrix))

    # Cálculo dos gradientes
    delta_end = h - y_matrix
    deltas = [delta_end]
    fraction = 1/m
    J_grads = [fraction * delta_end.T.dot(activations[-2])]
    for i in range(len(layer_size)-2, 0, -1):
        _, derivada = activation_function(nets[i])
        Theta = Thetas[i]
        delta = deltas[0]
        deltas = [np.matmul(delta, Theta[:, 1:]) * derivada] + \
            deltas  # list concatenation

        delta = deltas[0]
        a = activations[i-1]
        J_grads = [fraction * delta.T.dot(a)] + J_grads  # list concatenation

    grad = np.array([])
    for i in range(len(layer_size)-1):
        grad = np.concatenate((grad, J_grads[i].ravel()), axis=0)

    return J, grad


def randInitializeWeights(L_in, L_out, epsilon_init=0):
    """
    Randomly initialize the weights of a layer in a neural network
    """

    if epsilon_init == 0:
        epsilon_init = np.sqrt(6)/np.sqrt(L_in + L_out)

    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W


def gradient(i, learning_rate, theta, X_train, X_val, y_train, y_val, layer_size, activation_function):
    """Gradient descent"""
    J_history, J_hist_val = np.zeros(i), np.zeros(i)
    theta_history = np.zeros((i, theta.shape[0]))
    acc_hist_train, acc_hist_val = np.zeros(i), np.zeros(i)
    
    m = y.shape[0]
    num_labels = np.unique(y).shape[0]

    J, grad = nn_cost_function(theta,
                             num_labels,
                             X_train, y_train, activation_function=activation_function, layer_size=layer_size)

    for epoch in range(i):
        theta = theta - learning_rate * grad
        
        acc_train = predict(X_train, y_train, layer_size, theta, activation_function, 'Training')
        acc_val = predict(X_val, y_val, layer_size, theta, activation_function, 'Validation')
        
        J, grad = nn_cost_function(theta,
                                 num_labels,
                                 X_train, y_train, activation_function=activation_function, layer_size=layer_size)
        
        J_val, _ = nn_cost_function(theta,
                                 num_labels,
                                 X_val, y_val, activation_function=activation_function, layer_size=layer_size)

        theta_history[epoch, :] = theta.ravel()
        J_history[epoch] = J
        J_hist_val[epoch] = J_val
        acc_hist_train[epoch] = acc_train
        acc_hist_val[epoch] = acc_val

    return theta, theta_history, J_history, J_hist_val, acc_hist_train, acc_hist_val


def run_neural_network(X, y, layer_size, activation_function='sigmoid', epochs=500, learning_rate=1, training_set=0.80, validation_set=0.10):
    """Function that runs the neural network implemented to solve the problem"""
    
    # Divide in training, validation and test
    m = y.shape[0]
    
    rng = np.random.default_rng()
    idx = rng.permutation(m) # Shuffle
    
    X = X[idx, :]
    y = y[idx]
    
    # Divide
    n_train = np.round(m*training_set).astype(int)
    n_val = np.round(m*validation_set).astype(int)
        
    X_train = X[:n_train, :]
    X_val = X[n_train:n_train+n_val, :]
    X_test = X[n_train+n_val:, :]
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]
    
    # Set activation function
    activation_function = set_activation_function(activation_function)
    
    # Thetas unrolled
    initial_nn_params = np.array([])
    for i in range(len(layer_size)-1):
        initial_nn_params = np.concatenate((initial_nn_params, randInitializeWeights(
            layer_size[i], layer_size[i+1]).ravel()), axis=0)
    
    # Check neural network gradients
    # check_NN_gradients(nn_cost_function, activation_function)
    
    # Gradient descent
    theta, theta_history, J_hist_train, J_hist_val, acc_hist_train, acc_hist_val = gradient(
        epochs, learning_rate, initial_nn_params, X_train, X_val, y_train, y_val, layer_size, activation_function
        )
    
    # Test prediction
    predict(X_test, y_test, layer_size, theta, activation_function, 'Test', True)
    print('Training accuracy', acc_hist_train[-1])
    print('Validation accuracy', acc_hist_val[-1])

    
    # Plot accuracy (training and validation)
    pyplot.rcParams.update({'font.size': 35})
    pyplot.figure(figsize=(16, 11))
    pyplot.plot(np.arange(1, epochs+1), acc_hist_train, 'b-o', linewidth=2)
    pyplot.plot(np.arange(1, epochs+1), acc_hist_val, 'r-o', linewidth=2)
    pyplot.xlabel('Épocas')
    pyplot.ylabel('Acurácia')
    pyplot.legend(['Acurácia de treinamento', 'Acurácia de validação'])
    pyplot.grid(visible=True, which='both')
    pyplot.yticks(ticks=np.arange(0, 101, 10))
    pyplot.title(f'Taxa de aprendizado: {learning_rate}')
    pyplot.savefig(f'accuracy_{J_hist_train.shape[0]}epochs_{learning_rate}alpha.png', dpi=300)
    
    # Plot loss (training and validation)
    pyplot.figure(figsize=(16, 11))
    pyplot.plot(np.arange(1, epochs+1), J_hist_train, 'b-o', linewidth=2)
    pyplot.plot(np.arange(1, epochs+1), J_hist_val, 'r-o', linewidth=2)
    pyplot.xlabel('Épocas')
    pyplot.ylabel('Custo')
    pyplot.legend(['Custo de treinamento', 'Custo de validação'])   
    pyplot.grid(visible=True, which='both')
    pyplot.title(f'Taxa de aprendizado: {learning_rate}')
    pyplot.savefig(f'cost_{J_hist_train.shape[0]}epochs_{learning_rate}alpha.png', dpi=300)
    
    return theta


# Problema 2
data = np.loadtxt('Trabalho prático 1 - Backpropagation/classification2.txt', delimiter=',')
X, y = data[:,:-1], data[:,-1].ravel().astype(int)

# Problema 3
# data = scipy.io.loadmat('Trabalho prático 1 - Backpropagation/classification3.mat')
# X, y = data['X'], data['y'].ravel()
# y[y == 10] = 0

# Inputs
layer_size = [X.shape[1], 8, 8, np.unique(y).shape[0]]
activation_function = 'sigmoid'
training_set = 0.80
validation_set = 0.10
epochs = 500
learning_rate = 2

theta = run_neural_network(X, y, layer_size, activation_function, epochs, learning_rate, training_set, validation_set)



# -*- coding: utf-8 -*-
"""Código extra - trabalho backprop

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XTtZGgpAefbiWejTrEjsnWzS_XXYdzff
"""

###############
##### Plotando fronteira de decisão não-linear
###############

import matplotlib.pyplot as plt

if X.shape[1] == 2:
    # Plotando fronteira de decisão
    x1s = np.linspace(-1,1.5,200)
    x2s = np.linspace(-1,1.5,200)
    z=np.zeros((len(x1s),len(x2s)))


    # Set activation function
    activation_function = set_activation_function(activation_function)

    for i in range(len(x1s)):
        for j in range(len(x2s)):
            # x = np.array([x1s[i], x2s[j]]).reshape(2,-1)
            # z[i,j] = net_z_output( x )  # saida do modelo antes de aplicar a função sigmoide - substituir aqui teu código
            x = np.array([[x1s[i], x2s[j]]])
            _, nets = all_activations(x, layer_size, theta, activation_function)
            zzz = nets[-1]
            z[i,j] = np.sum(zzz)
    pyplot.rcParams.update({'font.size': 48})
    plt.figure(figsize=(16,9))
    plt.contour(x1s,x2s,z.T,0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc=0)

    ###############
    ##### Classificação binária com modelo de rede neural - backprop / regressão logística
    ###############

    import pandas as pd
    df=pd.read_csv("Trabalho prático 1 - Backpropagation/classification2.txt", header=None)

    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["Accepted","Rejected"],loc=0)

if X.shape[1] > 2:
    ###############
    ##### Classificação de dígitos com modelo de rede neural - backprop
    ###############

    from scipy.io import loadmat
    mat=loadmat("Trabalho prático 1 - Backpropagation/classification3.mat")
    X=mat["X"]
    y=mat["y"]

    import matplotlib.image as mpimg
    fig, axis = plt.subplots(10,10,figsize=(8,8))
    for i in range(10):
        for j in range(10):
            axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
            axis[i,j].axis("off")

print('FIM')