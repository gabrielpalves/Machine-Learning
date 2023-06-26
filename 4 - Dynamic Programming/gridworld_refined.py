import numpy as np
import matplotlib.pyplot as plt


def take_action(state, action, size):
    """An action causes a state transition and returns a reward"""
    
    # terminal state
    if np.all(state == np.zeros((2))) or np.all(state == np.ones((2))*(size-1)):
        return state, 0

    state_transition = state + action
    
    # state remains unchanged if out of bounds
    if state_transition[0] < 0 or state_transition[0] > size-1: # up and down bounds
        state_transition = state
    
    elif state_transition[1] < 0 or state_transition[1] > size-1: # left and right bounds
        state_transition = state
        
    return state_transition, -1


def plot_gridworld(states_history, steps, Delta_history):
    for k in steps:
        plt.figure(k)
        Delta = Delta_history[k]
        states = states_history[:,:,k]
        # Plot grid
        rows = states.shape[0]
        cols = states.shape[1]
        for coord in range(rows+1):
            plt.plot([coord, coord], [0, rows], linewidth=2.5, c='k')
            plt.plot([0, rows], [coord, coord], linewidth=2.5, c='k')

        plt.title(f'k = {k}, Δ = {Delta:.4f}')

        # Plot values
        for row in range(states.shape[0]):
            for col in range(states.shape[1]):
                if (row != 0 and col != cols-1) or (row != rows-1 and col != 0):
                    plt.text(col+0.5, rows-row-0.5, f'{row*rows + col}\n{states[row, col]:.1f}', horizontalalignment='center')

        # Terminal states
        plt.fill_between([0, 1], cols-1, cols, color='gray')
        plt.fill_between([rows-1, rows], 0, 1, color='gray')
        
        plt.xticks(np.arange(0, cols+1))
        plt.yticks(np.arange(0, rows+1))
        
        plt.savefig(f'k{k}.png')

"""
Iterative police evaluation
applies the same operation to each state s: it replaces the old value of s with a new value
obtained from the old values of the successor states of s, and the expected immediate
rewards, along all the one-step transitions possible under the policy being evaluated. We
call this kind of operation an expected update.

Iterative Policy Evaluation, for estimating V = vπ

Input π, the policy to be evaluated
Algorithm parameter: a small threshold theta > 0 determining accuracy of estimation
Initialize V(s) arbitrarily, for s in S, and V (terminal) to 0
Loop:
  Delta <- 0
  Loop for each s in S:
      v <- V(s)
      V(s) <- np.sum(policy) * np.sum(probability) * (reward + discount_rate*value2)
      Delta <- max(Delta, |v - V(s)|)
until Delta < theta
"""

if __name__ == '__main__':
    size = 4 # size of gridworld (matrix of size x size)
    policy = 0.25 # always the same (25% chance of selecting each possible action)
    actions = np.array([
        [0, -1], # left
        [0, 1], # right
        [-1, 0], # up
        [1, 0] # down
    ], dtype='int8')
    theta = 0.01 # parameter for convergence
    discount = 1 # for no discount, discount must = 1
    
    vk0 = np.zeros((size, size), dtype='float32') # initial values for all states
    vk1 = np.zeros((size, size), dtype='float32')
    
    Delta = np.inf
    Delta_history = np.zeros((1))
    k = 0
    states_history = np.zeros((size, size, 1))
    # loop until convergence (Delta < theta)
    while Delta >= theta:
        Delta = 0
        # loop for each state
        for i in range(size*size):
            state = np.array([i/size, i%size], dtype='int8')
            
            # compute the value function
            v = 0
            for act_idx in range(actions.shape[0]):
                action = actions[act_idx, :]
                state_transition, reward = take_action(state, action, size)
                v += policy * (reward + discount * vk0[state_transition[0], state_transition[1]])
            vk1[state[0], state[1]] = v
            # calculate convergence
            Delta = np.max([Delta, np.abs(vk0[state[0], state[1]] - vk1[state[0], state[1]])])
        
        # update old state
        vk0 = np.copy(vk1)
        k += 1
        
        # save history
        states_history = np.concatenate([states_history, vk0.reshape(size, size, 1)], axis=2)
        Delta_history = np.concatenate([Delta_history, Delta], axis=None)

    print(k, Delta)
    print(vk0)

    steps = [0, 1, 2, 3, 10, k]
    plot_gridworld(states_history, steps, Delta_history)
    plt.show()
    