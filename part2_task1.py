"""
Part 2 – Task 1: MDP Planning with a Known Transition Model
===========================================================
Implements two classical planning algorithms on a 5×5 stochastic grid world:
    (a) Value Iteration  – Bellman optimality equation iterated to convergence.
    (b) Policy Iteration – Alternates between policy evaluation and policy
                          improvement until the policy is stable.

Grid World
----------
    • Size      : 5 × 5, states (x, y) with x, y ∈ {0,1,2,3,4}
    • Start     : (0, 0) – bottom-left corner
    • Goal      : (4, 4) – top-right corner  (terminal state)
    • Roadblocks: (2, 1) and (2, 3)         (impassable cells)
    • Actions   : U (up), D (down), L (left), R (right)

Stochastic Transition Model (known to the agent in Task 1)
-----------------------------------------------------------
    When action a is chosen in state s:
        • Prob 0.8 : move in intended direction
        • Prob 0.1 : move perpendicular-left  of intended direction
        • Prob 0.1 : move perpendicular-right of intended direction
    If a resulting move would leave the grid or enter a roadblock, the
    agent stays in its current cell.

Rewards
-------
    • −1 for every step taken
    • +10 upon entering the goal state (then episode terminates)

Discount factor γ = 0.9
"""

# ============================================================
# Environment Constants
# ============================================================

GRID_SIZE   = 5
START       = (0, 0)
GOAL        = (4, 4)
ROADBLOCKS  = {(2, 1), (2, 3)}
GAMMA       = 0.9

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_SYMBOLS = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', None: 'G'}

# Perpendicular directions for each intended action
#   (perpendicular-left, perpendicular-right)
PERP = {
    'U': ('L', 'R'),
    'D': ('R', 'L'),   # left of "Down" is Right; right of "Down" is Left
    'L': ('D', 'U'),
    'R': ('U', 'D'),
}

# ============================================================
# Environment Helpers
# ============================================================

def all_states():
    """All valid (non-roadblock) grid cells."""
    return [(x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in ROADBLOCKS]


def move(state, action):
    """
    Apply a *deterministic* step from state in the given direction.
    Returns the resulting state; stays in place if the move is invalid.
    """
    x, y = state
    if   action == 'U': nx, ny = x,     y + 1
    elif action == 'D': nx, ny = x,     y - 1
    elif action == 'L': nx, ny = x - 1, y
    elif action == 'R': nx, ny = x + 1, y
    else: raise ValueError(f"Unknown action: {action}")

    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in ROADBLOCKS:
        return (nx, ny)
    return state   # wall or roadblock → stay


def transitions(state, action):
    """
    Full stochastic transition model: T(s, a) → list of (s', prob, reward).

    The reward is:
        +10   if s' == GOAL
        −1    otherwise
    Probabilities for outcomes that share the same next-state are merged.
    """
    if state == GOAL:
        return []   # absorbing terminal state

    perp_l, perp_r = PERP[action]
    raw = [(action, 0.8), (perp_l, 0.1), (perp_r, 0.1)]

    # Merge probabilities for identical next-states (e.g. two perp moves
    # both blocked → both stay; they sum to the same s').
    outcomes = {}
    for act, prob in raw:
        ns = move(state, act)
        reward = 10.0 if ns == GOAL else -1.0
        if ns not in outcomes:
            outcomes[ns] = [0.0, reward]
        outcomes[ns][0] += prob

    return [(ns, p, r) for ns, (p, r) in outcomes.items()]

# ============================================================
# (a) Value Iteration
# ============================================================

def value_iteration(theta=1e-9):
    """
    Value Iteration (Bellman Optimality Operator).

    Iteratively applies:
        V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V_k(s') ]

    until max |V_{k+1}(s) − V_k(s)| < theta.

    Parameters
    ----------
    theta : convergence threshold (default 1e-9)

    Returns
    -------
    V      : dict  {state → optimal value}
    policy : dict  {state → optimal action}
    """
    states = all_states()
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        delta   = 0.0
        new_V   = {}

        for s in states:
            if s == GOAL:
                new_V[s] = 0.0   # terminal state
                continue

            # Bellman optimality backup
            new_V[s] = max(
                sum(prob * (r + GAMMA * V[ns])
                    for ns, prob, r in transitions(s, a))
                for a in ACTIONS
            )
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        iteration += 1

        if delta < theta:
            print(f"  Value Iteration converged after {iteration} sweeps "
                  f"(Δ = {delta:.2e})")
            break

    # Greedy policy extraction
    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = None
            continue
        policy[s] = max(
            ACTIONS,
            key=lambda a: sum(prob * (r + GAMMA * V[ns])
                              for ns, prob, r in transitions(s, a))
        )

    return V, policy

# ============================================================
# (b) Policy Iteration
# ============================================================

def policy_evaluation(policy, theta=1e-9):
    """
    Iterative Policy Evaluation (Bellman Expectation Equation).

    Solves:
        V^π_{k+1}(s) = Σ_{s'} P(s'|s, π(s)) [ R(s,π(s),s') + γ V^π_k(s') ]

    until convergence within theta.

    Parameters
    ----------
    policy : dict {state → action}
    theta  : convergence threshold

    Returns
    -------
    V : dict {state → value under the given policy}
    """
    states = all_states()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        new_V = {}

        for s in states:
            if s == GOAL or policy[s] is None:
                new_V[s] = 0.0
                continue
            a = policy[s]
            new_V[s] = sum(
                prob * (r + GAMMA * V[ns])
                for ns, prob, r in transitions(s, a)
            )
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            break

    return V


def policy_iteration():
    """
    Policy Iteration.

    Alternates between:
        1. Policy Evaluation  – compute V^π for the current policy.
        2. Policy Improvement – update π greedily with respect to V^π.

    Terminates when the policy no longer changes (guaranteed to converge
    to the optimal policy for finite MDPs).

    Returns
    -------
    V      : dict {state → optimal value}
    policy : dict {state → optimal action}
    """
    states = all_states()

    # Initialise with a fixed (arbitrary) policy: always go Up
    policy = {s: 'U' for s in states}
    policy[GOAL] = None

    iteration = 0
    while True:
        # --- Step 1: Policy Evaluation ---
        V = policy_evaluation(policy)

        # --- Step 2: Policy Improvement ---
        policy_stable = True
        for s in states:
            if s == GOAL:
                continue
            old_a = policy[s]
            best_a = max(
                ACTIONS,
                key=lambda a: sum(prob * (r + GAMMA * V[ns])
                                  for ns, prob, r in transitions(s, a))
            )
            policy[s] = best_a
            if best_a != old_a:
                policy_stable = False

        iteration += 1
        if policy_stable:
            print(f"  Policy Iteration converged after {iteration} "
                  f"evaluation/improvement cycles")
            break

    return V, policy

# ============================================================
# Display Helpers
# ============================================================

def print_value_function(V, title="Value Function"):
    """Print V as a grid, high-y rows first (y=4 at top)."""
    print(f"\n  {title}")
    print("  " + "-" * 41)
    header = "       " + "".join(f"  x={x}  " for x in range(GRID_SIZE))
    print(header)
    for y in range(GRID_SIZE - 1, -1, -1):
        row = f"  y={y} |"
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "  BLK  "
            elif s == GOAL:
                row += " [GOAL]"
            else:
                row += f" {V.get(s, 0.0):+.2f} "
        print(row)
    print()


def print_policy(policy, title="Policy"):
    """Print policy as a grid of arrows."""
    print(f"\n  {title}")
    print("  " + "-" * 41)
    header = "       " + "".join(f"  x={x} " for x in range(GRID_SIZE))
    print(header)
    for y in range(GRID_SIZE - 1, -1, -1):
        row = f"  y={y} |"
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "   B  "
            else:
                sym = ACTION_SYMBOLS.get(policy.get(s), '?')
                row += f"   {sym}  "
        print(row)
    print()


def compare_policies(policy_vi, policy_pi, label_a="VI", label_b="PI"):
    """Report states where two policies differ."""
    diffs = [s for s in all_states()
             if s != GOAL and policy_vi.get(s) != policy_pi.get(s)]
    if not diffs:
        print(f"  ✓ Policies from {label_a} and {label_b} are IDENTICAL.\n")
    else:
        print(f"  Policies differ at {len(diffs)} state(s):")
        for s in diffs:
            print(f"    {s}: {label_a}={policy_vi[s]}  {label_b}={policy_pi[s]}")
        print()


def max_value_diff(V1, V2):
    """Maximum absolute difference between two value functions."""
    states = all_states()
    return max(abs(V1[s] - V2.get(s, 0.0)) for s in states)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PART 2 – TASK 1: MDP Planning (Known Transition Model)")
    print("  Grid: 5×5  |  γ = 0.9  |  Stochastic (0.8 / 0.1 / 0.1)")
    print("=" * 60)

    # ------ Value Iteration ------
    print("\n[Value Iteration]")
    V_vi, policy_vi = value_iteration()
    print_value_function(V_vi, "Optimal Value Function  (Value Iteration)")
    print_policy(policy_vi,    "Optimal Policy          (Value Iteration)")

    # ------ Policy Iteration ------
    print("\n[Policy Iteration]")
    V_pi, policy_pi = policy_iteration()
    print_value_function(V_pi, "Optimal Value Function  (Policy Iteration)")
    print_policy(policy_pi,    "Optimal Policy          (Policy Iteration)")

    # ------ Comparison ------
    print("\n[Comparison: Value Iteration vs Policy Iteration]")
    compare_policies(policy_vi, policy_pi)
    diff = max_value_diff(V_vi, V_pi)
    print(f"  Max |V_VI(s) − V_PI(s)| across all states: {diff:.2e}")
    print()
