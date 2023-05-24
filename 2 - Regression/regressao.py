print("Author: Gabriel Padilha Alves") # Author: Gabriel Padilha Alves

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm


def vector2matrix(x):
    """Transform a vector of n elements in a matrix of n x 1"""
    return np.array([x]).T if len(x.shape) == 1 else x


def f_true(x):
    # return 2 + 0.8 * x
    y = np.ones((x.shape[0]))*2
    for i in range(x.shape[1]):
        xc = np.exp(x[:,i])
        y = y + 0.4 * (i + 1)/2 * xc
    return y


def normal_equations(x, y):
    """Computes the closed-form solution to linear regression"""
    return np.matmul(np.matmul( inv(np.matmul(x.T, x)), x.T ), y)


def map_feature(x, degree):
    """Maps x to a polynomial of a desired degree"""
    if degree <= 1:
        return x
    
    if x.shape[1] == 2:
        x0 = x[:, 0]
        x1 = x[:, 1]
        
        out = np.ones((x1.shape[0], np.sum(np.arange(degree+2))-1))
        count = 0
        for i in range(1, degree+1):
            for j in range(i+1):
                out[:, count] = x0**(i-j) * x1**j
                count += 1
    elif x.shape[1] == 1:
        out = np.ones((x.shape[0], degree))
        count = 0
        x0 = x[:,0]
        for i in range(1, degree+1):
            out[:, count] = x0**i
            count += 1
    else:
        return x
    
    return out


def normalize(x):
    """Normalize feature"""
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = (x - mu)/sigma
    return x


def h(x, theta):
    """hypothesis

    Args:
        x (numpy array): data of the problem
        theta (numpy array): parameters
    """
    return vector2matrix(np.matmul(x, theta))


def J(theta, xs, ys):
    """Cost function

    Args:
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    return 1/2/ys.shape[0] * np.sum( (h(xs, theta) - ys)**2 )


def gradient(i, learning_rate, theta, x_ori, xs, ys):
    """Gradient descent

    Args:
        alpha (float): learning rate
        i (int): number of epochs
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    
    Returns:
        theta (numpy array): optimum parameters
        theta_history (numpy array): parameters history
        J_history (numpy array): history of the cost
    """
    J_history = np.zeros(i)
    theta_history = np.zeros((i, theta.shape[0]))
    fig, ax = plt.subplots()
    plotted = True
    alpha_init = 0.2
    c = 'r'
    label = 'Regression progression'
    for epoch in range(i):
        # Parameters (Theta)
        m = ys.shape[0]
        theta = theta - ( learning_rate/m ) * ( np.matmul( xs.T, h(xs, theta) - ys ) )
        
        # Save variables
        theta_history[epoch, :] = theta.ravel()
        J_history[epoch] = J(theta, xs, ys)
        
        # Plot
        alpha = alpha_init + 0.5*(epoch/i)**2
        if epoch == i-1: alpha, c, plotted, label = 1, 'g', False, 'Final regression line'
        print_modelo(theta, x_ori, xs, ys, fig, ax, plotted, alpha, c=c, label=label)
        label = None
            
    # Save regression plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.rcParams.update({'font.size': 24})
    fig.set_size_inches(16, 9)
    fig.savefig(f'regression_{learning_rate}alpha_{i}epochs.png', dpi=300)
    return theta, theta_history, J_history


def print_modelo(theta, x_original, xs, ys, fig, ax, plotted, alpha=1, c='r', label=None):
    """Plot on the same graph:
    - the model/hypothesis (line)
    - the original line (true function)
    - and the data with noise
    
    Args:
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    x = vector2matrix(xs[:, 1])
    y = f_true(x_original) # true function
    yr = h(xs, theta) # regression
    
    if not plotted:
        # Scatter original data
        ax.scatter(x, ys, linewidths=2.5, c='b', marker='+', label='Data')
        
        # Plot original line
        ax.plot(x, y, linewidth=2.5, c='k', label='True function')
    
    # Plot regression line
    ax.plot(x, yr, linewidth=2.5, linestyle='--', c=c, alpha=alpha, label=label)
    
    
def print_results(theta_hist, J_hist, xs, ys, learning_rate):
    # Plot gradient convergence (epoch x J)
    fig = plt.figure(2)
    plt.plot(np.arange(J_hist.shape[0]), J_hist, '-o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Cost (J)')
    plt.rcParams.update({'font.size': 24})
    fig.set_size_inches(16, 9)
    fig.savefig(f'cost_{J_hist.shape[0]}epochs_{learning_rate}alpha.png', dpi=300)
    
    if xs.shape[1] > 2:
        return
    
    # Now prepare to plot the cost in function of theta_0 and theta_1
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    
    theta0, theta1 = theta_hist[:,0], theta_hist[:,1]
    t0 = np.arange(-5, 5, 0.01)
    t1 = np.copy(t0)
    t0, t1 = np.meshgrid(t0, t1)
        
    y = np.zeros((t0.shape))
    for i in range(t0.shape[0]):
        for j in range(t1.shape[0]):
            t = np.array([t0[i,j], t1[i,j]])
            y[i,j] = J(t, xs, ys)
            
    # Plot surface of the cost function (theta_0 x theta_1 x J)
    ax1.plot_surface(t0, t1, y, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.rcParams.update({'font.size': 24})
    
    # Plot gradient convergence (theta_0 x theta_1 x J)
    ax2.contour(t0, t1, y)
    ax2.plot3D(theta0, theta1, J_hist, '-ro', linewidth=2.5)
    ax2.view_init(elev=30, azim=45)
    
    fig.set_size_inches(16, 9)
    fig.savefig(f'theta_and_cost_{J_hist.shape[0]}epochs_{learning_rate}alpha.png', dpi=300)


if __name__ == '__main__':
    # Data set {(x,y)}
    m = 100
    learning_rate = 0.01
    polynomial = True
    multivariable = True
    normalize_data = True
    
    xs = np.linspace(-3, 3, m)
    if multivariable:
        xs = np.array([np.linspace(-3, 3, m), np.linspace(.3, -1, m)]).T
    xs = vector2matrix(xs)
    x_original = xs
    ys = f_true(xs)
    ys = vector2matrix(ys) + np.random.randn(ys.shape[0], 1)*0.5
    
    # Polynomial regression
    if polynomial:
        xs = map_feature(xs, degree=4)
    
    # Normalize data
    if normalize_data:
        xs = normalize(xs)

    # Add ones because of theta_0
    xs = np.concatenate((np.ones((xs.shape[0], 1)), xs), axis=1)
    
    # Initial theta
    theta_init = vector2matrix(np.zeros(xs.shape[1]))
    
    theta, theta_hist, J_hist = gradient(i=5000, learning_rate=learning_rate, theta=theta_init, x_ori=x_original, xs=xs, ys=ys)
    print_results(theta_hist, J_hist, xs, ys, learning_rate)
    print(f'Theta found with gradient descent: {theta}')
    try: print(f'Theta found with the closed-form solution: {normal_equations(xs, ys)}')
    except: pass
