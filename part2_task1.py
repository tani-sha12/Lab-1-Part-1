grid_size = 5
start = (0, 0)
goal = (4, 4)
roadblocks = {(2, 1), (2, 3)}
gamma = 0.9

actions = ['U', 'D', 'L', 'R']
action_symbol = {'U': '^', 'D': 'v', 'L': '<', 'R': '>', None: 'G'}

# Perpendicular directions for each intended action
# (perpendicular-left, perpendicular-right)
perpendicular = {
    'U': ('L', 'R'),
    'D': ('R', 'L'),   # left of "Down" is Right; right of "Down" is Left
    'L': ('D', 'U'),
    'R': ('U', 'D'),
}

def all_states():
    """All valid (non-roadblock) grid cells."""
    return [(x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if (x, y) not in roadblocks]


def move(state, action):
    """
    Apply a *deterministic* step from state in the given direction.
    Returns the resulting state; stays in place if the move is invalid.
    """
    x, y = state
    if action == 'U': 
        nx, ny = x, y + 1
    elif action == 'D': 
        nx, ny = x, y - 1
    elif action == 'L': 
        nx, ny = x - 1, y
    elif action == 'R': 
        nx, ny = x + 1, y
    else: 
        raise ValueError(f"Unknown action:{action}")

    if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in roadblocks:
        return (nx, ny)
    return state   # wall or roadblock → stay


def transitions(state, action):
    """
    Full stochastic transition model: T(s, a) → list of (s', prob, reward).

    The reward is:
        +10   if s' == GOAL
        -1    otherwise
    Probabilities for outcomes that share the same next-state are merged.
    """
    if state == goal:
        return []

    perp_l, perp_r = perpendicular[action]
    raw = [(action, 0.8), (perp_l, 0.1), (perp_r, 0.1)] # Merge probabilities for identical next-states
    outcomes = {}
    for act, prob in raw:
        ns = move(state, act)
        reward = 10.0 if ns == goal else -1.0
        if ns not in outcomes:
            outcomes[ns] = [0.0, reward]
        outcomes[ns][0] += prob

    return [(ns,p,r) for ns, (p,r) in outcomes.items()]

def value_iteration(theta=1e-9):
    states = all_states()
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        delta = 0.0
        new_V = {}

        for s in states:
            if s == goal:
                new_V[s] = 0.0 # terminal state
                continue

            # Bellman optimality backup
            new_V[s] = max(
                sum(prob * (r + gamma * V[ns])
                    for ns, prob, r in transitions(s, a))
                for a in actions
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
        if s == goal:
            policy[s] = None
            continue
        policy[s] = max(
            actions,
            key=lambda a: sum(prob * (r + gamma * V[ns])
                              for ns, prob, r in transitions(s, a))
        )

    return V, policy

def policy_evaluation(policy, theta=1e-9):
    states = all_states()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        new_V = {}

        for s in states:
            if s == goal or policy[s] is None:
                new_V[s] = 0.0
                continue
            a = policy[s]
            new_V[s] = sum(
                prob * (r + gamma * V[ns])
                for ns, prob, r in transitions(s, a)
            )
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            break

    return V


def policy_iteration():
    states = all_states()
    # Initialise with a fixed (arbitrary) policy: always go Up
    policy = {s: 'U' for s in states}
    policy[goal] = None

    iteration = 0
    while True:
        V = policy_evaluation(policy)
        policy_stable = True
        for s in states:
            if s == goal:
                continue
            old_a = policy[s]
            best_a = max(
                actions,
                key=lambda a: sum(prob * (r + gamma * V[ns])
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

def print_value_function(V, title="Value Function"):
    """Print V as a grid, high-y rows first (y=4 at top)."""
    print(f"\n  {title}")
    print("  " + "-" * 41)
    header = "       " + "".join(f"  x={x}  " for x in range(grid_size))
    print(header)
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
    header = "       " + "".join(f"  x={x} " for x in range(grid_size))
    print(header)
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


def compare_policies(policy_vi, policy_pi, label_a="VI", label_b="PI"):
    #report states where two policies differ
    diffs = [s for s in all_states()
             if s != goal and policy_vi.get(s) != policy_pi.get(s)]
    if not diffs:
        print(f"  Policies from {label_a} and {label_b} are identical.\n")
    else:
        print(f"  Policies differ at {len(diffs)} state(s):")
        for s in diffs:
            print(f"    {s}: {label_a}={policy_vi[s]}  {label_b}={policy_pi[s]}")
        print()


def max_value_diff(V1, V2):
    """Maximum absolute difference between two value functions."""
    states = all_states()
    return max(abs(V1[s] - V2.get(s, 0.0)) for s in states)

if __name__ == "__main__":
    print("=" * 60)
    print("PART 2 - TASK 1: MDP Planning (Known Transition Model)")
    print("  Grid: 5x5  |  γ = 0.9  |  Stochastic (0.8 / 0.1 / 0.1)")
    print("=" * 60)

    print("\n[Value Iteration]")
    V_vi, policy_vi = value_iteration()
    print_value_function(V_vi, "Optimal Value Function  (Value Iteration)")
    print_policy(policy_vi,    "Optimal Policy          (Value Iteration)")

    print("\n[Policy Iteration]")
    V_pi, policy_pi = policy_iteration()
    print_value_function(V_pi, "Optimal Value Function  (Policy Iteration)")
    print_policy(policy_pi,    "Optimal Policy          (Policy Iteration)")

    print("\n[Comparison: Value Iteration vs Policy Iteration]")
    compare_policies(policy_vi, policy_pi)
    diff = max_value_diff(V_vi, V_pi)
    print(f"  Max |V_VI(s) − V_PI(s)| across all states: {diff:.2e}")
    print()
