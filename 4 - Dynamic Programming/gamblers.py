"""
Policy Iteration (using iterative policy evaluation) for estimating π ≈ π*
1. Initialization
    V(s) ∈ R and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
    Loop:
        Δ <- 0
        Loop for each s ∈ S:
            v <- V(s)
            V(s) <- Σ_{s', r} p(s', r | s, π(s)) [r + γV(s')]
            Δ <- max(Δ, |v - V(s)|)
    until Δ < θ (a small positive number determining the accuracy of estimation)

3. Policy Improvement
    policy-stable <- true
    For each s ∈ S:
        old-action π(s)
        π(s) <- argmax_a Σ_{s', r} p(s', r | s, a) [r + γV(s')]
        If old-action ≠ π(s), then policy-stable <- false
    If policy-stable, then stop and return V ≈ v* and π ≈ π*; else go to 2
    
Policy iteration requires multiple sweeps through the state set
just for the one application of policy evaluation
            |
            v
Truncate policy evaluation
(stop before convergence)
            |
            v
Value Iteration limits policy evaluation to only one sweep.
Combining policy improvement and truncated policy evaluation into a simple update:

v_{k+1}(s) = max_a Σ_{s', r} p(s', r | s, a)[r + γv_k(s')], for all s ∈ S

For arbitrary v_0, the sequence {v_k} can be shown to converge to v*.
Basically, the max operation is added to the update of policy evaluation.

Value Iteration, for estimating π ≈ π*

Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation
Initialize V(s), for all s ∈ S+, arbitrarily except that V(terminal) = 0
Loop:
    Δ <- 0
    Loop for each s ∈ S:
        v <- V(s)
        V(s) <- max_a Σ_{s', r} p(s', r | s, a) [r + γV(s')]
        Δ <- max(Δ, |v - V(s)|)
until Δ < θ

Output a deterministic policy, π ≈ π*, such that
    π(s) = argmax_a Σ_{s', r} p(s', r | s, a) [r + γV(s')]
    (Bellman optimality equation)

Gambler's Problem - A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips
- if the coin comes up heads, he wins as many dollars as he has staked on that flip;
- if it is tails, he loses his stake

- the game ends when the gambler wins by reaching his goal of $100, or loses by running out of money
- on each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars

Reward is zero on all transitions, except those on which the gambler reaches his goal (+1)
"""

import numpy as np

GOAL = 100
LOSS = 0

# small threshold θ > 0 determining accuracy of estimation
DELTA = np.inf
THETA = 0.0001

# discount
GAMMA = 1 # γ = 1 for no discount

# $1 to 99$; 0 = loss, 100 = win
states = np.arange(LOSS, GOAL+1, dtype='int8') 

# probability of the coin coming up heads (heads -> agent wins its stake)
p_h = 0.40


def bet(state, stake):
    """
    Returns 0 of reward for all states and 1 when the agent achieves its goal
    """
    next_state = GAMMA * (state + stake)
    reward = 0
    
    if next_state == GOAL:
        reward = 1
    elif next_state == LOSS:
        next_state = 0 # resets
    
    return np.array([reward, next_state], dtype='int8')

# Value Iteration, for estimating π ≈ π*
v_k1 = np.zeros((GOAL+1-LOSS)) # 0 to 100 (0 = no money, i.e., LOSS and 100 = GOAL)
while DELTA > THETA:
    DELTA = 0
    for s in states:
        # bet from $0 to the quantity necessary to attain the GOAL
        actions = np.arange(np.min([s, GOAL-s]) + 1, dtype='int8')
        
        v_max = -np.inf
        for a in actions:
            value = p_h * np.sum(bet(s, a))
            if v_max < value:
                v_max = np.copy(value)
        DELTA = np.max([DELTA, np.abs(v_k1[s] - v_max)])
        v_k1[s] = v_max
    
print(v_k1)

# Output a deterministic policy, π ≈ π*, such that
# π(s) = argmax_a Σ_{s', r} p(s', r | s, a) [r + γV(s')]
policy = np.zeros((GOAL+1-LOSS))
for s in states:
    v_max, best_action = v_k1[s], -np.inf
    
    actions = np.arange(np.min([s, GOAL-s]) + 1, dtype='int8')
    for a in actions:
        value = p_h * np.sum(bet(s, a))
        if value > v_max:
            v_max = np.copy(value)
            best_action = np.copy(a)
    policy[s] = a
print(policy)