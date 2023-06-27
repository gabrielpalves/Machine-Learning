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
import matplotlib.pyplot as plt

GOAL = 100
LOSS = 0

# small threshold θ > 0 determining accuracy of estimation
DELTA = np.inf
THETA = 1e-8

# discount
GAMMA = 1 # γ = 1 for no discount

# $1 to 99$; 0 = loss, 100 = win
STATES = np.arange(LOSS, GOAL+1, dtype='int8') 

# probability of the coin coming up heads (heads -> agent wins its stake)
P_H = 0.40


def bet(state, stake, result=1):
    """
    Returns 0 of reward for all states and 1 when the agent achieves its goal
    """
    next_state = state + stake*result
    reward = 0
    
    if next_state == GOAL:
        reward = 1
    
    return next_state, reward


# Value Iteration, for estimating π ≈ π*
policy = np.zeros((GOAL+1-LOSS))
v_k1 = np.zeros((GOAL+1-LOSS)) # 0 to 100 (0 = no money, i.e., LOSS and 100 = GOAL)
v_k1[GOAL] = 1
while DELTA >= THETA:
    DELTA = 0
    for s in STATES[1:GOAL]:
        v_k0 = v_k1.copy()
        # bet from $0 to the quantity necessary to attain the GOAL
        actions = np.arange(np.min([s, GOAL-s]) + 1, dtype='int8')
        
        v_max = -np.inf
        for a in actions:
            next_s, r = bet(s, a, 1)
            value = P_H * (r + GAMMA*v_k1[next_s])
            
            next_s, r = bet(s, a, -1)
            value += (1 - P_H) * (r + GAMMA*v_k1[next_s])
            
            if v_max < value:
                v_max = np.copy(value)
            
        v_k1[s] = v_max
        DELTA = np.max([DELTA, np.abs(v_k0[s] - v_k1[s])])


# Output a deterministic policy, π ≈ π*, such that
# π(s) = argmax_a Σ_{s', r} p(s', r | s, a) [r + γV(s')]
policy = np.zeros(GOAL + 1)
for s in STATES[1:GOAL]:
    # bet from $0 to the quantity necessary to attain the GOAL
    actions = np.arange(np.min([s, GOAL-s]) + 1, dtype='int8')
    
    v_max, best_action = -np.inf, -np.inf
    for a in actions:
        next_s, r = bet(s, a, 1)
        value = P_H * (r + GAMMA*v_k1[next_s])
        
        next_s, r = bet(s, a, -1)
        value += (1 - P_H) * (r + GAMMA*v_k1[next_s])
        
        if v_max < value:
            v_max = np.copy(value)
            best_action = np.copy(a)
    policy[s] = best_action
    
print(v_k1)
print(policy)

plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(16, 9))
plt.plot(STATES, v_k1)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.savefig('value_estimates.png', dpi=300)

plt.figure(figsize=(16, 9))
plt.bar(STATES, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.savefig('final_policy.png', dpi=300)

plt.show()
