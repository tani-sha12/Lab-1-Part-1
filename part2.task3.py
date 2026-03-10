"""
Part 2 – Task 3: Tabular Q-Learning (Model-Free, Off-Policy TD)
================================================================
Like Task 2, the agent does NOT know the transition probabilities.
The environment still follows the same stochastic dynamics (0.8/0.1/0.1).

Instead of waiting for complete episodes (Monte Carlo), the agent
performs online, step-by-step updates using the Q-learning rule
(a special case of temporal-difference learning, TD(0)):

    Q(s, a) ← Q(s, a) + α [ r + γ max_{a'} Q(s', a') − Q(s, a) ]

Key hyperparameters:
    • ε = 0.1  (fixed ε-greedy exploration)
    • α = 0.1  (fixed learning rate)
    • γ = 0.9  (discount factor)

After training, the learned Q-learning policy is compared against:
    • The optimal policy from Task 1 (Value / Policy Iteration)
    • The Monte Carlo policy from Task 2

Grid World (same as Tasks 1 & 2)
----------------------------------
    5×5 grid, Start (0,0), Goal (4,4), Roadblocks {(2,1),(2,3)}
    Rewards: −1 per step, +10 on reaching GOAL
"""

import random
from collections import defaultdict

# ============================================================
# Environment Constants
# ============================================================

GRID_SIZE    = 5
START        = (0, 0)
GOAL         = (4, 4)
ROADBLOCKS   = {(2, 1), (2, 3)}
GAMMA        = 0.9
EPSILON      = 0.1
ALPHA        = 0.1          # learning rate
NUM_EPISODES = 50_000
MAX_STEPS    = 500          # safety cap per episode

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_SYMBOLS = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', None: 'G'}

PERP = {
    'U': ('L', 'R'),
    'D': ('R', 'L'),
    'L': ('D', 'U'),
    'R': ('U', 'D'),
}

# ============================================================
# Environment Dynamics (unknown to agent; used only to simulate)
# ============================================================

def move(state, action):
    """Deterministic move; stays in place if wall or roadblock."""
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
    Stochastic transition (environment-side; not accessible to the agent).
        intended direction  → prob 0.8
        perpendicular-left  → prob 0.1
        perpendicular-right → prob 0.1

    Returns (next_state, reward).
    """
    perp_l, perp_r = PERP[action]
    outcomes = [(action, 0.8), (perp_l, 0.1), (perp_r, 0.1)]

    r_val      = random.random()
    cumulative = 0.0
    for act, prob in outcomes:
        cumulative += prob
        if r_val <= cumulative:
            ns     = move(state, act)
            reward = 10.0 if ns == GOAL else -1.0
            return ns, reward

    ns = move(state, action)
    return ns, (10.0 if ns == GOAL else -1.0)

# ============================================================
# ε-Greedy Action Selection
# ============================================================

def epsilon_greedy(Q, state, epsilon):
    """
    ε-greedy action selection.
        With prob ε      → random action (exploration).
        With prob 1 − ε  → greedy action w.r.t. Q(state, ·),
                           ties broken uniformly at random.
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_vals = [Q[(state, a)] for a in ACTIONS]
    max_q  = max(q_vals)
    best   = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
    return random.choice(best)

# ============================================================
# Tabular Q-Learning
# ============================================================

def q_learning(num_episodes=NUM_EPISODES,
               epsilon=EPSILON,
               alpha=ALPHA,
               gamma=GAMMA,
               seed=42):
    """
    Tabular Q-Learning (off-policy TD control).

    Algorithm (one episode)
    -----------------------
        Initialise s ← START
        Repeat (until s == GOAL or max_steps reached):
            a  ← ε-greedy(Q, s)            # behaviour policy
            s', r ← env.step(s, a)          # stochastic environment
            Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]
            s  ← s'

    Key property: the update target uses max_{a'} Q(s',a'), making this
    an *off-policy* algorithm — the agent learns the value of the
    *greedy* target policy regardless of which ε-greedy action it took.

    Parameters
    ----------
    num_episodes : total training episodes
    epsilon      : exploration rate (fixed at 0.1)
    alpha        : step size / learning rate (fixed at 0.1)
    gamma        : discount factor (0.9)
    seed         : random seed for reproducibility

    Returns
    -------
    Q      : dict {(state, action) → estimated action-value}
    policy : dict {state → greedy action w.r.t. Q}
    rewards_per_episode : list of total undiscounted rewards per episode
                          (useful for plotting convergence)
    """
    random.seed(seed)
    Q = defaultdict(float)           # all Q-values initialise to 0
    rewards_per_episode = []

    for ep in range(1, num_episodes + 1):
        state        = START
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            # --- Action selection (ε-greedy behaviour policy) ---
            action = epsilon_greedy(Q, state, epsilon)

            # --- Interact with stochastic environment ---
            next_state, reward = stochastic_step(state, action)
            total_reward      += reward

            # --- Q-learning update (off-policy TD(0)) ---
            max_next_q     = max(Q[(next_state, a)] for a in ACTIONS)
            td_target      = reward + gamma * max_next_q
            td_error       = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            if state == GOAL:
                break

        rewards_per_episode.append(total_reward)

        if ep % 10_000 == 0:
            avg_reward = sum(rewards_per_episode[-10_000:]) / 10_000
            print(f"    Episodes: {ep}/{num_episodes}  |  "
                  f"Avg reward (last 10k): {avg_reward:.2f}")

    # Extract deterministic greedy policy
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

    return Q, policy, rewards_per_episode

# ============================================================
# Display Helpers
# ============================================================

def all_states():
    return [(x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in ROADBLOCKS]


def q_to_V(Q):
    """V(s) = max_a Q(s, a)."""
    V = {}
    for s in all_states():
        V[s] = 0.0 if s == GOAL else max(Q[(s, a)] for a in ACTIONS)
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


def compare_policies(policy_a, policy_b, label_a="A", label_b="B"):
    """Quantify agreement between two policies over non-terminal states."""
    states     = [s for s in all_states() if s != GOAL]
    matches    = sum(1 for s in states if policy_a.get(s) == policy_b.get(s))
    total      = len(states)
    mismatches = [(s, policy_a.get(s), policy_b.get(s))
                  for s in states if policy_a.get(s) != policy_b.get(s)]
    print(f"  {label_a} vs {label_b}:")
    print(f"    Agreement: {matches}/{total} states ({100*matches/total:.1f}%)")
    if mismatches:
        for s, a, b in mismatches:
            print(f"      State {s}: {label_a}={a}  {label_b}={b}")
    print()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from part2_task1 import value_iteration
    from part2_task2 import monte_carlo_control

    print("=" * 60)
    print("PART 2 – TASK 3: Tabular Q-Learning (Model-Free, Off-Policy)")
    print(f"  Episodes: {NUM_EPISODES}  |  ε = {EPSILON}  |"
          f"  α = {ALPHA}  |  γ = {GAMMA}")
    print("=" * 60)

    # --- Reference policies ---
    print("\n[Loading Task 1 optimal policy...]")
    _, policy_vi = value_iteration()

    print(f"\n[Training Monte Carlo agent ({NUM_EPISODES} episodes) ...]")
    _, policy_mc = monte_carlo_control(num_episodes=NUM_EPISODES)

    # --- Q-Learning Training ---
    print(f"\n[Training Q-Learning agent ({NUM_EPISODES} episodes) ...]")
    Q_ql, policy_ql, ep_rewards = q_learning()

    # --- Results ---
    V_ql = q_to_V(Q_ql)
    print_value_function(V_ql, "State-Value Function  V(s) = max_a Q(s,a)  [Q-Learning]")
    print_policy(policy_ql, "Learned Policy (Q-Learning)")

    print_policy(policy_mc, "Reference: Learned Policy (Monte Carlo – Task 2)")
    print_policy(policy_vi, "Reference: Optimal Policy (Value Iteration – Task 1)")

    # --- Policy Comparisons ---
    print("\n[Policy Comparisons]")
    compare_policies(policy_ql, policy_vi, label_a="Q-Learning", label_b="Optimal (VI)")
    compare_policies(policy_ql, policy_mc, label_a="Q-Learning", label_b="MC Control")
    compare_policies(policy_mc, policy_vi, label_a="MC Control", label_b="Optimal (VI)")

    # --- Convergence Summary ---
    chunk = 5_000
    print("\n[Q-Learning Convergence: Average Episode Reward]")
    for i in range(0, NUM_EPISODES, chunk):
        window = ep_rewards[i: i + chunk]
        avg    = sum(window) / len(window)
        print(f"    Episodes {i+1:>6}–{i+chunk:<6}: avg reward = {avg:+.2f}")
    print()
