import numpy as np

states = np.zeros((4, 4))
policy = np.array([0.25, 0.25, 0.25, 0.25]) # always the same (25% chance of selecting each possible action)
actions = np.array([
    [0, -1], # left
    [0, 1], # right
    [-1, 0], # up
    [1, 0] # down
], dtype='int16')
theta = 0.01
discount = 1 # for no discount, discount must = 1


def take_action(state, action):
    """An action causes a state transition and returns a reward"""
    
    # terminal state
    if np.all(state == np.zeros((2))) or np.all(state == np.ones((2))*3):
        return state, 0

    state_transition = state + action
    
    # state remains unchanged if out of bounds
    if state_transition[0] < 0 or state_transition[0] >= 3: # up and down bounds
        state_transition = state
    
    elif state_transition[1] < 0 or state_transition[1] >= 3: # left and right bounds
        state_transition = state
        
    return state_transition, -1


# Iterative police evaluation
# value = np.sum(policy) * np.sum(probability) * (reward + discount_rate*value2)

# iterative policy evaluation
# applies the same operation to each state s: it replaces the old value of s with a new value
# obtained from the old values of the successor states of s, and the expected immediate
# rewards, along all the one-step transitions possible under the policy being evaluated. We
# call this kind of operation an expected update.


# Iterative Policy Evaluation, for estimating V = vπ
# Input π, the policy to be evaluated
# Algorithm parameter: a small threshold theta > 0 determining accuracy of estimation
# Initialize V(s) arbitrarily, for s in S, and V (terminal) to 0
# Loop:
#   Delta <- 0
#   Loop for each s in S:
#       v <- V(s)
#       V(s) <- np.sum(policy) * np.sum(probability) * (reward + discount_rate*value2)
#       Delta <- max(Delta, |v − V(s)|)
# until Delta < theta

Delta = np.inf
k = 0
V = np.empty_like(states)
while Delta > theta:
    Delta = 0
    k += 1
    # loop for each state
    for i in range(4*4):
        state = np.array([i/4, i%4], dtype='int16')
        
        # compute the value function
        v = 0
        for act_idx in range(actions.shape[0]):
            action = actions[act_idx, :]
            state_transition, reward = take_action(state, action)
            v += policy[act_idx] * (reward + discount * states[state_transition[0], state_transition[1]])
        Delta = np.max([Delta, np.abs(v - V[state[0], state[1]])])
        V[state[0], state[1]] = v * discount
    states = np.copy(V)

print(k, Delta)
print(states)