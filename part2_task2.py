"""
Part 2 – Task 2: Monte Carlo Control (Model-Free)
==================================================
The agent does NOT know the transition probabilities.
The environment, however, still follows the same stochastic dynamics
(0.8 / 0.1 / 0.1) as in Task 1.

The agent learns by interacting with the environment through complete
episodes (Monte Carlo), using:
    • First-Visit MC prediction to estimate Q(s, a)
    • ε-greedy policy improvement  (ε = 0.1, fixed throughout)

After training, the learned policy is extracted and printed for
comparison with the optimal policies from Task 1.

Grid World (same as Task 1)
----------------------------
    5×5 grid, Start (0,0), Goal (4,4), Roadblocks {(2,1),(2,3)}
    Stochastic transitions: 0.8 intended, 0.1 each perpendicular
    Rewards: −1 per step, +10 on reaching GOAL
    Discount factor γ = 0.9
"""

import random
from collections import defaultdict

# ============================================================
# Environment Constants
# ============================================================

GRID_SIZE  = 5
START      = (0, 0)
GOAL       = (4, 4)
ROADBLOCKS = {(2, 1), (2, 3)}
GAMMA      = 0.9
EPSILON    = 0.1
NUM_EPISODES = 50_000

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_SYMBOLS = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', None: 'G'}

PERP = {
    'U': ('L', 'R'),
    'D': ('R', 'L'),
    'L': ('D', 'U'),
    'R': ('U', 'D'),
}

# ============================================================
# Environment Dynamics (embedded in the environment, unknown to agent)
# ============================================================

def move(state, action):
    """Deterministic move; stays if wall or roadblock."""
    x, y = state
    if   action == 'U': nx, ny = x,     y + 1
    elif action == 'D': nx, ny = x,     y - 1
    elif action == 'L': nx, ny = x - 1, y
    elif action == 'R': nx, ny = x + 1, y
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in ROADBLOCKS:
        return (nx, ny)
    return state


def stochastic_step(state, action):
    """
    Stochastic transition model (part of the environment, NOT the agent).
    The agent only sees the resulting next_state and reward; it cannot
    access the probabilities directly.

        intended direction  → prob 0.8
        perpendicular-left  → prob 0.1
        perpendicular-right → prob 0.1

    Returns (next_state, reward).
    """
    perp_l, perp_r = PERP[action]
    outcomes = [(action, 0.8), (perp_l, 0.1), (perp_r, 0.1)]

    r_val = random.random()
    cumulative = 0.0
    for act, prob in outcomes:
        cumulative += prob
        if r_val <= cumulative:
            ns = move(state, act)
            reward = 10.0 if ns == GOAL else -1.0
            return ns, reward

    # Fallback (floating-point edge case)
    ns = move(state, action)
    return ns, (10.0 if ns == GOAL else -1.0)

# ============================================================
# ε-Greedy Policy
# ============================================================

def epsilon_greedy(Q, state, epsilon):
    """
    ε-greedy action selection over Q(state, ·).
        • With prob ε  : choose a random action uniformly.
        • With prob 1−ε: choose the greedy (highest Q-value) action.
          Ties are broken randomly among maximisers.
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_vals = [Q[(state, a)] for a in ACTIONS]
    max_q  = max(q_vals)
    best   = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
    return random.choice(best)

# ============================================================
# Episode Generation
# ============================================================

def generate_episode(Q, epsilon, max_steps=500):
    """
    Run one full episode from START to GOAL (or max_steps).

    Returns a list of (state, action, reward) triples.
    The agent selects actions via its current ε-greedy policy derived from Q.
    Transitions are sampled from the stochastic environment.
    """
    episode = []
    state   = START

    for _ in range(max_steps):
        action          = epsilon_greedy(Q, state, epsilon)
        next_state, r   = stochastic_step(state, action)
        episode.append((state, action, r))
        state = next_state
        if state == GOAL:
            break

    return episode

# ============================================================
# First-Visit Monte Carlo Control
# ============================================================

def monte_carlo_control(num_episodes=NUM_EPISODES,
                        epsilon=EPSILON,
                        gamma=GAMMA,
                        seed=42):
    """
    First-Visit Monte Carlo Control with ε-greedy exploration.

    Algorithm
    ---------
    For each episode:
        1. Generate episode using current ε-greedy policy.
        2. Compute discounted return  G_t = Σ_{k≥0} γ^k · r_{t+k+1}
           for each time step t (working backwards for efficiency).
        3. For each (s, a) that appears FIRST at time t:
               returns(s, a).append(G_t)
               Q(s, a) ← mean of returns(s, a)
        4. The policy implicitly improves because epsilon_greedy uses
           the updated Q.

    The use of *first-visit* MC means only the first occurrence of each
    (state, action) pair within an episode contributes to the return
    estimate.  This guarantees unbiased estimates under the behaviour
    policy.

    Parameters
    ----------
    num_episodes : number of training episodes
    epsilon      : fixed exploration rate (ε = 0.1)
    gamma        : discount factor (γ = 0.9)
    seed         : random seed for reproducibility

    Returns
    -------
    Q      : dict {(state, action) → estimated action-value}
    policy : dict {state → greedy action w.r.t. Q}
    """
    random.seed(seed)

    Q             = defaultdict(float)   # Q(s, a) initialised to 0
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(int)

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(Q, epsilon)

        # --- First-Visit MC Return Computation ---
        G       = 0.0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            sa = (state, action)
            if sa not in visited:          # first-visit check
                visited.add(sa)
                returns_sum[sa]   += G
                returns_count[sa] += 1
                Q[sa] = returns_sum[sa] / returns_count[sa]

        if ep % 10_000 == 0:
            print(f"    Episodes completed: {ep}/{num_episodes}")

    # Extract greedy (deterministic) policy from Q
    policy = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                continue
            if s == GOAL:
                policy[s] = None
                continue
            policy[s] = max(ACTIONS, key=lambda a: Q[(s, a)])

    return Q, policy

# ============================================================
# Display Helpers
# ============================================================

def all_states():
    return [(x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in ROADBLOCKS]


def q_to_V(Q):
    """Derive state-value function: V(s) = max_a Q(s,a)."""
    V = {}
    for s in all_states():
        if s == GOAL:
            V[s] = 0.0
        else:
            V[s] = max(Q[(s, a)] for a in ACTIONS)
    return V


def print_value_function(V, title="Value Function"):
    print(f"\n  {title}")
    print("  " + "-" * 41)
    print("       " + "".join(f"  x={x}  " for x in range(GRID_SIZE)))
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
    print(f"\n  {title}")
    print("  " + "-" * 41)
    print("       " + "".join(f"  x={x} " for x in range(GRID_SIZE)))
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


def compare_to_optimal(mc_policy, optimal_policy, label="MC vs Optimal"):
    """Report how many states the MC policy matches the optimal policy."""
    states      = [s for s in all_states() if s != GOAL]
    matches     = sum(1 for s in states if mc_policy.get(s) == optimal_policy.get(s))
    total       = len(states)
    mismatches  = [(s, mc_policy.get(s), optimal_policy.get(s))
                   for s in states if mc_policy.get(s) != optimal_policy.get(s)]
    print(f"  {label}:")
    print(f"    Agreement: {matches}/{total} states ({100*matches/total:.1f}%)")
    if mismatches:
        print(f"    Mismatches:")
        for s, mc_a, opt_a in mismatches:
            print(f"      State {s}: MC={mc_a}  Optimal={opt_a}")
    print()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Import Task 1 optimal policy for comparison
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from part2_task1 import value_iteration

    print("=" * 60)
    print("PART 2 – TASK 2: Monte Carlo Control (Model-Free)")
    print(f"  Episodes: {NUM_EPISODES}  |  ε = {EPSILON}  |  γ = {GAMMA}")
    print("=" * 60)

    # --- Get optimal policy from Task 1 for later comparison ---
    print("\n[Loading Task 1 optimal policy for comparison...]")
    _, optimal_policy = value_iteration()

    # --- Monte Carlo Training ---
    print(f"\n[Training Monte Carlo agent for {NUM_EPISODES} episodes...]")
    Q_mc, policy_mc = monte_carlo_control()

    # --- Display Results ---
    V_mc = q_to_V(Q_mc)
    print_value_function(V_mc, "State-Value Function  V(s) = max_a Q(s,a)  [MC]")
    print_policy(policy_mc,    "Learned Policy (Monte Carlo)")
    print_policy(optimal_policy, "Reference: Optimal Policy (Task 1 – Value Iteration)")

    # --- Policy Comparison ---
    print("\n[Policy Comparison]")
    compare_to_optimal(policy_mc, optimal_policy, label="MC Control vs Optimal (VI)")
