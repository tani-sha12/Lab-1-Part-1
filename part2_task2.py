import random
from collections import defaultdict

grid_size  = 5
start      = (0, 0)
goal       = (4, 4)
roadblocks = {(2, 1), (2, 3)}
GAMMA      = 0.9
epsilon    = 0.1
num_episodes = 50_000

actions = ['U', 'D', 'L', 'R']
action_symbol = {'U': '^', 'D': 'v', 'L': '<', 'R': '>', None: 'G'}

perpendicular = {
    'U': ('L', 'R'),
    'D': ('R', 'L'),
    'L': ('D', 'U'),
    'R': ('U', 'D'),
}

def move(state, action):
    """Deterministic move; stays if wall or roadblock."""
    x, y = state
    if action == 'U': nx, ny = x, y + 1
    elif action == 'D': nx, ny = x, y - 1
    elif action == 'L': nx, ny = x - 1, y
    elif action == 'R': nx, ny = x + 1, y
    if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in roadblocks:
        return (nx, ny)
    return state


def stochastic_step(state, action):
    perp_l, perp_r = perpendicular[action]
    outcomes = [(action, 0.8), (perp_l, 0.1), (perp_r, 0.1)]

    r_val = random.random()
    cumulative = 0.0
    for act, prob in outcomes:
        cumulative += prob
        if r_val <= cumulative:
            ns = move(state, act)
            if ns == goal:
                reward = 10.0 
            else: 
                reward = -1.0
            return ns, reward

    # Fallback (floating-point edge case)
    ns = move(state, action)
    return ns, (10.0 if ns == goal else -1.0)

def epsilon_greedy(Q, state, epsilon):
    """
    ε-greedy action selection over Q(state, ·).
        • With prob ε  : choose a random action uniformly.
        • With prob 1-ε: choose the greedy (highest Q-value) action.
          Ties are broken randomly among maximisers.
    """
    if random.random() < epsilon:
        return random.choice(actions)
    q_vals = [Q[(state, a)] for a in actions]
    max_q = max(q_vals)
    best = [a for a, q in zip(actions, q_vals) if q == max_q]
    return random.choice(best)

def generate_episode(Q, epsilon, max_steps=500):
    """
    Run one full episode from START to GOAL (or max_steps).
    Returns a list of (state, action, reward) triples.
    The agent selects actions via its current ε-greedy policy derived from Q.
    Transitions are sampled from the stochastic environment.
    """
    episode = []
    state = start

    for x in range(max_steps):
        action = epsilon_greedy(Q, state, epsilon)
        next_state, r = stochastic_step(state, action)
        episode.append((state, action, r))
        state = next_state
        if state == goal:
            break

    return episode

def monte_carlo_control(num_episodes=num_episodes, epsilon=epsilon,gamma=GAMMA,seed=42):
    random.seed(seed)

    Q = defaultdict(float) # Q(s, a) initialised to 0
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(int)

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(Q, epsilon)
        G = 0.0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            sa = (state, action)
            if sa not in visited: # first-visit check
                visited.add(sa)
                returns_sum[sa]   += G
                returns_count[sa] += 1
                Q[sa] = returns_sum[sa] / returns_count[sa]

        if ep % 10_000 == 0:
            print(f"    Episodes completed: {ep}/{num_episodes}")

    # Extract greedy (deterministic) policy from Q
    policy = {}
    for x in range(grid_size):
        for y in range(grid_size):
            s = (x, y)
            if s in roadblocks:
                continue
            if s == goal:
                policy[s] = None
                continue
            policy[s] = max(actions, key=lambda a: Q[(s, a)])

    return Q, policy

def all_states():
    return[(x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if (x, y) not in roadblocks]


def q_to_V(Q):
    """Derive state-value function: V(s) = max_a Q(s,a)."""
    V = {}
    for s in all_states():
        if s == goal:
            V[s] = 0.0
        else:
            V[s] = max(Q[(s, a)] for a in actions)
    return V


def print_value_function(V, title="Value Function"):
    print(f"\n  {title}")
    print("  " + "-" * 41)
    print("       " + "".join(f"  x={x}  " for x in range(grid_size)))
    for y in range(grid_size - 1, -1, -1):
        row = f"  y={y} |"
        for x in range(grid_size):
            s = (x, y)
            if s in roadblocks:
                row += "  BLK  "
            elif s == goal:
                row += " [GOAL]"
            else:
                row += f" {V.get(s, 0.0):+.2f} "
        print(row)
    print()


def print_policy(policy, title="Policy"):
    print(f"\n  {title}")
    print("  " + "-" * 41)
    print("       " + "".join(f"  x={x} " for x in range(grid_size)))
    for y in range(grid_size - 1, -1, -1):
        row = f"  y={y} |"
        for x in range(grid_size):
            s = (x, y)
            if s in roadblocks:
                row += "   B  "
            else:
                sym = action_symbol.get(policy.get(s), '?')
                row += f"   {sym}  "
        print(row)
    print()


def compare_to_optimal(mc_policy, optimal_policy, label="MC vs Optimal"):
    """Report how many states the MC policy matches the optimal policy."""
    states = [s for s in all_states() if s != goal]
    matches = sum(1 for s in states if mc_policy.get(s) == optimal_policy.get(s))
    total = len(states)
    mismatches = [(s, mc_policy.get(s), optimal_policy.get(s))
                   for s in states if mc_policy.get(s) != optimal_policy.get(s)]
    print(f"  {label}:")
    print(f"    Agreement: {matches}/{total} states ({100*matches/total:.1f}%)")
    if mismatches:
        print(f"    Mismatches:")
        for s, mc_a, opt_a in mismatches:
            print(f"      State{s}: MC={mc_a}  Optimal={opt_a}")
    print()

if __name__ == "__main__":
    # Import Task 1 optimal policy for comparison
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from part2_task1 import value_iteration

    print("=" * 60)
    print("PART 2 TASK 2: Monte Carlo Control (Model-Free)")
    print(f"  Episodes: {num_episodes}  |  ε = {epsilon}  |  γ = {GAMMA}")
    print("=" * 60)

    print("[Loading Task 1 optimal policy for comparison...]")
    _, optimal_policy = value_iteration()

    print(f"[Training Monte Carlo agent for {num_episodes} episodes...]")
    Q_mc, policy_mc = monte_carlo_control()

    V_mc = q_to_V(Q_mc)
    print_value_function(V_mc, "State-Value Function  V(s) = max_a Q(s,a) [Monte Carlo]")
    print_policy(policy_mc,    "Learned Policy(Monte Carlo)")
    print_policy(optimal_policy, "Reference: Optimal Policy (Task 1 – Value Iteration)")

    print("[Policy Comparison]")
    compare_to_optimal(policy_mc, optimal_policy, label="MC Control vs Optimal (Value Iteration)")
