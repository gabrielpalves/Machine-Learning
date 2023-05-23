print("Author: Gabriel Padilha Alves") # Author: Gabriel Padilha Alves

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm






def f_true(x):
    return 2 + 0.8 * x


def normal_equations(x, y):
    """Computes the closed-form solution to linear regression"""
    if len(y.shape) == 1: y = np.array([y]).T
    theta = np.matmul(np.matmul( inv(np.matmul(x.T, x)), x.T ), y)
    return theta


def map_feature(x, degree):
    """Maps x to a polynomial"""
    x0 = x[:, 0]
    x1 = x[:, 1]
    
    out = np.ones(x1.shape[0], np.sum(np.arange(degree+2)))
    count = 1
    for i in range(1, degree+1):
        for j in range(i):
            out[:, count] = (x0**(i-j)) * (x1**j)
            count += 1
    return out


def h(x, theta):
    """hypothesis

    Args:
        x (numpy array): data of the problem
        theta (numpy array): parameters
    """
    return np.matmul(x, theta)


def J(theta, xs, ys):
    """Cost function

    Args:
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    return 1/2/ys.shape[0] * np.sum( (h(xs, theta) - ys)**2 )


def gradient(i, learning_rate, theta, xs, ys):
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
    plotted = False
    alpha = 0.15
    for epoch in range(i):
        # Parameters (Theta)
        m = ys.shape[0]
        theta = theta - ( learning_rate/m ) * ( np.matmul(xs.T, h(xs, theta) - ys) )
        
        # Save variables
        theta_history[epoch, :] = theta
        J_history[epoch] = J(theta, xs, ys)
        
        # Plot
        if (
            epoch == 0 or
            epoch == i-1 or
            np.abs(theta_history[epoch-1, 0] - theta[0])/theta[0] > 0.1 or
            np.abs(theta_history[epoch-1, 1] - theta[1])/theta[1] > 0.1
            ):
            alpha = alpha + 0.05
            if alpha > 1: alpha = 1
            if epoch == i-1: alpha = 1
            print_modelo(theta, xs, ys, fig, ax, plotted, alpha) 
            plotted = True
            
    # Save regression plot
    plt.xlabel('x')
    plt.ylabel('y')
    fig.set_size_inches(16, 9)
    fig.savefig(f'regression_{learning_rate}alpha_{i}epochs.png', dpi=300)
    return theta, theta_history, J_history


def print_modelo(theta, xs, ys, fig, ax, plotted, alpha=1):
    """Plot on the same graph:
    - the model/hypothesis (line)
    - the original line (true function)
    - and the data with noise
    
    Args:
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    x = xs[:, 1]
    y = f_true(x) # true function
    yr = h(xs, theta) # regression
    
    if not plotted:
        # Scatter original data
        ax.scatter(x, ys, linewidths=2.5, c='b', marker='+')
        
        # Plot original line
        ax.plot(x, y, linewidth=2.5, c='k')
    
    # Plot regression line
    ax.plot(x, yr, linewidth=2.5, linestyle='--', c='r', alpha=alpha)
    
    
def print_results(theta_hist, J_hist, xs, ys, learning_rate):
    # Plot gradient convergence (epoch x J)
    fig = plt.figure(2)
    plt.plot(np.arange(J_hist.shape[0]), J_hist, '-o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Cost (J)')
    fig.set_size_inches(16, 9)
    fig.savefig(f'cost_{J_hist.shape[0]}epochs_{learning_rate}alpha.png', dpi=300)
    
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
    
    # Plot gradient convergence (theta_0 x theta_1 x J)
    ax2.contour(t0, t1, y)
    ax2.plot3D(theta0, theta1, J_hist, '-ro', linewidth=2.5)
    ax2.view_init(elev=30, azim=45)
    
    fig.set_size_inches(16, 9)
    fig.savefig(f'theta_and_cost_{J_hist.shape[0]}epochs_{learning_rate}alpha.png', dpi=300)


if __name__ == '__main__':
    # Data set {(x,y)}
    m = 100
    xs = np.linspace(-3, 3, m)
    ys = np.array( [f_true(x) + np.random.randn()*0.5 for x in xs] )

    # Normalize data
    mu = np.mean(xs, axis=0)
    sigma = np.std(xs, axis=0)
    xs = (xs - mu)/sigma

    # Add ones because of theta_0
    if len(xs.shape) == 1: xs = np.array([xs]).T
    xs = np.concatenate((np.ones((xs.shape[0], 1)), xs), axis=1)
    
    # Initial theta
    theta_init = np.zeros(xs.shape[1])-1
    # Learning rate
    learning_rate = 0.01
    
    theta, theta_hist, J_hist = gradient(i=5000, learning_rate=learning_rate, theta=theta_init, xs=xs, ys=ys)
    print_results(theta_hist, J_hist, xs, ys, learning_rate)
    print(f'Theta found with the closed-form solution: {normal_equations(xs, ys)}')
    print(f'Theta found with gradient descent: {theta}')
