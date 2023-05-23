print("Author: Gabriel Padilha Alves") # Author: Gabriel Padilha Alves

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def f_true(x):
    return 2 + 0.8 * x

# conjunto de dados {(x,y)}
xs = np.linspace(-3, 3, 100)
ys = np.array( [f_true(x) + np.random.randn()*0.5 for x in xs] )

def h(x, theta):
    """hypothesis

    Args:
        x (numpy array): data of the problem
        theta (numpy array): parameters
    """
    return np.matmul(np.array([np.ones(x.shape), x]).T, theta)


def J(theta, xs, ys):
    """Cost function

    Args:
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    return 1/2/ys.shape[0] * np.sum( (h(xs, theta) - ys)**2 )


def gradient(i, alpha, theta, xs, ys):
    """Gradient descent

    Args:
        alpha (float): learning rate
        i (int): number of epochs
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    J_history = np.zeros(i)
    theta_history = np.zeros((i, theta.shape[0]))
    fig, ax = plt.subplots()
    plotted = False
    for epoch in range(i):
        # Parameters
        theta = theta - ( alpha/ys.shape[0] )*( np.dot(xs, h(xs, theta) - ys) )
        
        # Save variables
        theta_history[epoch, :] = theta
        J_history[epoch] = J(theta, xs, ys)
        
        # Plot
        if epoch % int(i/100) == 0:
            print_modelo(theta, xs, ys, fig, ax, plotted)
            plotted = True
    return theta, theta_history, J_history


def print_modelo(theta, xs, ys, fig, ax, plotted):
    """Plot on the same graph:
    - the model/hypothesis (line)
    - the original line (true function)
    - and the data with noise
    
    Args:
        theta (numpy array): parameters
        xs (numpy array): input data
        ys (numpy array): output data
    """
    y = f_true(xs) # true function
    yr = h(xs, theta) # regression
    
    if not plotted:
        # Scatter original data
        ax.scatter(xs, ys, linewidths=2.5, c='b', marker='+')
        
        # Plot original line
        ax.plot(xs, y, linewidth=2.5, c='k')
    
    # Plot regression line
    ax.plot(xs, yr, linewidth=2.5, linestyle='--', c='r')
    
    plt.show()

def print_results(theta, J, xs, ys):
    # Plot gradient convergence (epoch x J)
    plt.figure(2)
    plt.plot(np.arange(J.shape[0]), J, linewidth=2.5)
    
    # Plot gradient convergence (theta_0 x theta_1 x J)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    theta0, theta1 = theta[:,0], theta[:,1]
    t0 = np.arange(-1, 1, 0.1)
    t1 = np.copy(t0)
    t0, t1 = np.meshgrid(t0, t1)
    
    y = np.zeros((t0.shape[0], t1.shape[0]))
    for i in range(t0.shape[0]):
        for j in range(t1.shape[0]):
            y[i,j] = h(xs, np.array((t0, t1), axis=1))
    
    surf = ax1.plot_surface(t0, t1, y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
            
            
    
    
    
    # Plot surface of (theta_0 x theta_1 x J)
    plt.figure(4)
    
    pass

theta, theta_hist, J_hist = gradient(i=5000, alpha=0.01, theta=np.zeros(2), xs=xs, ys=ys)


# print(theta)
# print(theta_hist)
# print(J_hist)