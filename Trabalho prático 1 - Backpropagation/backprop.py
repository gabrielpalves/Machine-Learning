import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize


def Theta_transform(layer_size, nn_params):
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

    # for i in range(len(Thetas)):
    #     print(Thetas[i].shape)
    
    return Thetas


def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        h = ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                      cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p


def debugInitializeWeights(fan_out, fan_in):
    """
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
    connections using a fixed strategy. This will help you later in debugging.

    Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
    the "bias" terms.

    Parameters
    ----------
    fan_out : int
        The number of outgoing connections.

    fan_in : int
        The number of incoming connections.

    Returns
    -------
    W : array_like (1+fan_in, fan_out)
        The initialized weights array given the dimensions.
    """
    # Initialize W using "sin". This ensures that W is always of the same values and will be
    # useful for debugging
    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def computeNumericalGradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.

    Parameters
    ----------
    J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
         those given parameters.

    e : float (optional)
        The value to use for epsilon for computing the finite difference.

    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad


def checkNNGradients(nnCostFunction, lambda_=0):
    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result in
    very similar values.

    Parameters
    ----------
    nnCostFunction : func
        A reference to the cost function implemented by the student.

    lambda_ : float (optional)
        The regularization parameter value.
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
    # print(y)
    # Unroll parameters
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
    print(nn_params.shape)

    # short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lambda_, layer_size=[3, 5, 3])
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(numgrad.shape, grad.shape)
    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)


def sigmoid(z):
    """Sigmoide"""
    s = lambda z: 1.0 / (1.0 + np.exp(-z))
    return s(z), s(z) * (1 - s(z))

def sigmoidGradient(z):
    """Derivada da função sigmoide"""
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    """Tangente hiperbólica"""
    th = lambda z: (np.exp(2*z) - 1)/(np.exp(2*z) + 1)
    return th(z), 1.0 - np.power(th(z), 2)

def tanh_gradient(z):
    """Derivada da tangente hiperbólica"""
    return 1 - np.power(tanh(z), 2)

import scipy.io
data = scipy.io.loadmat('Trabalho prático 1 - Backpropagation/ex4data1.mat')

# training data stored in arrays X, y
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
y[y == 10] = 0

# Number of training examples
m = y.size

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the weights into variables Theta1 and Theta2
weights = scipy.io.loadmat('Trabalho prático 1 - Backpropagation/ex4weights.mat')

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll parameters 
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

layer_size = [400, 25, 10]
activation_function = sigmoid


def all_activations(X, layer_size, nn_params, activation_function):
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
                
    return activations


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0, activation_function=sigmoid, layer_size=[400, 25, 10]):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 
    
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.
    
    input_layer_size : int
        Number of features for the input layer. 
    
    hidden_layer_size : int
        Number of hidden units in the second layer.
    
    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 
    
    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).
    
    y : array_like
        Dataset labels. A vector of shape (m,).
    
    lambda_ : float, optional
        Regularization parameter.
 
    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.
    
    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenation of
        neural network weights Theta1 and Theta2.
    
    Instructions
    ------------
    You should complete the code by working through the following parts.
    
    - Part 1: Feedforward the neural network and return the cost in the 
              variable J. After implementing Part 1, you can verify that your
              cost function computation is correct by verifying the cost
              computed in the following cell.
    
    - Part 2: Implement the backpropagation algorithm to compute the gradients
              Theta1_grad and Theta2_grad. You should return the partial derivatives of
              the cost function with respect to Theta1 and Theta2 in Theta1_grad and
              Theta2_grad, respectively. After implementing Part 2, you can check
              that your implementation is correct by running checkNNGradients provided
              in the utils.py module.
    
              Note: The vector y passed into the function is a vector of labels
                    containing values from 0..K-1. You need to map this vector into a 
                    binary vector of 1's and 0's to be used with the neural network
                    cost function.
     
              Hint: We recommend implementing backpropagation using a for-loop
                    over the training examples if you are implementing it for the 
                    first time.
    
    - Part 3: Implement regularization with the cost function and gradients.
    
              Hint: You can implement this around the code for
                    backpropagation. That is, you can compute the gradients for
                    the regularization separately and then add them to Theta1_grad
                    and Theta2_grad from Part 2.
    
    Note 
    ----
    We have provided an implementation for the sigmoid function in the file 
    `utils.py` accompanying this assignment.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network    
    Thetas = Theta_transform(layer_size, nn_params)
    
    # Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
    #                     (hidden_layer_size, (input_layer_size + 1)))

    # Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
    #                     (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size
         
    # You need to return the following variables correctly 
    J = 0
    # Theta1_grad = np.zeros(Theta1.shape)
    # Theta2_grad = np.zeros(Theta2.shape)
    
    # ====================== YOUR CODE HERE ======================
    
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

    # a1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # z2 = a1.dot(Theta1.T)
    # a2 = sigmoid(z2)

    # a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

    # z3 = a2.dot(Theta2.T)
    # a3 = sigmoid(z3)

    # Cálculo do custo J
    y_matrix = np.eye(num_labels)[y]

    h = activations[-1] # hipótese
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
        deltas = [np.matmul(delta, Theta[:,1:]) * derivada] + deltas # list concatenation
        
        delta = deltas[0]
        a = activations[i-1]
        J_grads = [fraction * delta.T.dot(a)] + J_grads # list concatenation
    
    grad = np.array([])
    for i in range(len(layer_size)-1):
        grad = np.concatenate((grad, J_grads[i].ravel()), axis=0)
    
    # delta_3 = h - y_matrix
    # delta_2 = np.matmul(delta_3,Theta2[:,1:]) * sigmoidGradient(z2)

    # Theta1_grad = (1 / m) * delta_2.T.dot(a1)
    # Theta2_grad = (1 / m) * delta_3.T.dot(a2)
    
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad



lambda_ = 0
cost, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_)
print('Cost at parameters (loaded from ex4weights): %.6f ' % cost)
print('The cost should be about                   : 0.287629.')



def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Parameters
    ----------
    L_in : int
        Number of incomming connections.
    
    L_out : int
        Number of outgoing connections. 
    
    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.
    
    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
        
    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds 
    to the parameters for the bias unit.
    """

    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================

    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    # ============================================================
    return W

# Todos os thetas em um só vetor
initial_nn_params = np.array([])
for i in range(len(layer_size)-1):
    initial_nn_params = np.concatenate((initial_nn_params, randInitializeWeights(layer_size[i], layer_size[i+1]).ravel()), axis=0)
print(initial_nn_params.shape)

print('Initializing Neural Network Parameters ...')
    
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
print(initial_nn_params.shape, 25*401+10*26, initial_Theta1.shape, initial_Theta2.shape)

checkNNGradients(nnCostFunction)

activations = all_activations(X, layer_size, nn_params, activation_function)
h = activations[-1]

def J(y, h):
    """Cost function

    Args:
        theta (numpy array): parameters
        x (numpy array): input data
        y (numpy array): output data
    """
    num_labels = np.unique(y).shape[0]
    y_matrix = np.eye(num_labels)[y]
    return (-1 / m) * np.sum((np.log(h) * y_matrix) + np.log(1 - h) * (1 - y_matrix))

print(y.shape, X.shape)

def gradient(i, learning_rate, theta, x, y, layer_size, activation_function):
    """Gradient descent

    Args:
        alpha (float): learning rate
        i (int): number of epochs
        theta (numpy array): parameters
        x (numpy array): input data
        y (numpy array): output data
    
    Returns:
        theta (numpy array): optimum parameters
        theta_history (numpy array): parameters history
        J_history (numpy array): history of the cost
    """
    
    J_history = np.zeros(i)
    theta_history = np.zeros((i, theta.shape[0]))
    
    activations = all_activations(x, layer_size, theta, activation_function)
    h = activations[-1]

    for epoch in range(i):
        # Parameters (Theta)
        m = y.shape[0]
        theta = theta - ( learning_rate/m ) * ( np.matmul( x.T, h - y ) )
        
        # Save variables
        activations = all_activations(x, layer_size, theta, activation_function)
        h = activations[-1]
        
        theta_history[epoch, :] = theta.ravel()
        J_history[epoch] = J(y, h)
    
    return theta, theta_history, J_history


gradient(10, 0.01, initial_nn_params, X, y, layer_size, activation_function)