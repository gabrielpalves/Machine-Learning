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

Gamber's Problem - A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips
- if the coin comes up heads, he wins as many dollars as he has staked on that flip;
- if it is tails, he loses his stake

- the game ends when the gambler wins by reaching his goal of $100, or loses by running out of money
- on each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars

Reward is zero on all transitions, except those on which the gambler reaches his goal (+1)
"""

victory = 100
defeat = 0

Delta = 0
theta = 0.01

state = 1

def bet(state, stake, result):
    next_state = state + stake*result, 0
    if next_state == victory:
        return next_state, 1
    elif next_state == defeat:
        return 1, 0
    else:
        return next_state, 0

