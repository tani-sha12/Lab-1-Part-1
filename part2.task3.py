import random
from collections import defaultdict

grid_size    = 5
start        = (0, 0)
goal         = (4, 4)
roadblocks   = {(2, 1), (2, 3)}
gamma        = 0.9
epsilon      = 0.1
alpha        = 0.1
num_episodes = 50_000
max_steps    = 500

actions = ['U', 'D', 'L', 'R']
action_symbol = {'U': '^', 'D': 'v', 'L': '<', 'R': '>', None: 'G'}

perpendicular = {
    'U': ('L', 'R'),   # perpendicular to Up → Left, Right
    'D': ('R', 'L'),   # perpendicular to Down → Right, Left
    'L': ('D', 'U'),   # perpendicular to Left → Down, Up
    'R': ('U', 'D'),   # perpendicular to Right → Up, Down
}

def move(state, action):
    """Deterministic move; stays in place if wall or roadblock."""
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

    ns = move(state, action)
    return ns, (10.0 if ns == goal else -1.0)

def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    q_vals = [Q[(state, a)] for a in actions]
    max_q = max(q_vals)
    best = [a for a, q in zip(actions, q_vals) if q == max_q]
    return random.choice(best)

def q_learning(num_episodes=num_episodes, epsilon=epsilon, alpha=alpha, gamma=gamma, seed=42):
    random.seed(seed)
    Q = defaultdict(float) # all Q-values initialise to 0
    rewards_per_episode = []

    for ep in range(1, num_episodes + 1):
        state = start
        total_reward = 0.0

        for x in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            next_state, reward = stochastic_step(state, action)
            total_reward += reward

            max_next_q = max(Q[(next_state, a)] for a in actions) #update q-learning
            td_target = reward + gamma * max_next_q
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            if state == goal:
                break

        rewards_per_episode.append(total_reward)

        if ep % 10_000 == 0:
            avg_reward = sum(rewards_per_episode[-10_000:]) / 10_000
            print(f"    Episodes: {ep}/{num_episodes}  |  "
                  f"Avg reward (last 10k): {avg_reward:.2f}")

    # Extract deterministic greedy policy
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

    return Q, policy, rewards_per_episode

def all_states():
    return [(x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if (x, y) not in roadblocks]

def q_to_V(Q):
    #V(s) = max_a Q(s, a)
    V = {}
    for s in all_states():
        V[s] = 0.0 if s == goal else max(Q[(s, a)] for a in actions)
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

def compare_policies(policy_a, policy_b, label_a="A", label_b="B"):
    """Quantify agreement between two policies over non-terminal states."""
    states = [s for s in all_states() if s != goal]
    matches = sum(1 for s in states if policy_a.get(s) == policy_b.get(s))
    total = len(states)
    mismatches = [(s, policy_a.get(s), policy_b.get(s))
                  for s in states if policy_a.get(s) != policy_b.get(s)]
    print(f"  {label_a} vs {label_b}:")
    print(f"    Agreement: {matches}/{total} states ({100*matches/total:.1f}%)")
    if mismatches:
        for s, a, b in mismatches:
            print(f"      State {s}: {label_a}={a}  {label_b}={b}")
    print()

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from part2_task1 import value_iteration
    from part2_task2 import monte_carlo_control

    print("=" * 60)
    print("PART 2 TASK 3: Tabular Q-Learning (Model-Free, Off-Policy)")
    print(f"  Episodes: {num_episodes}  |  ε = {epsilon}  |"
          f"  α = {alpha}  |  γ = {gamma}")
    print("=" * 60)

    print("\n[Loading Task 1 optimal policy...]")
    _, policy_vi = value_iteration()

    print(f"\n[Training Monte Carlo agent ({num_episodes} episodes) ...]")
    _, policy_mc = monte_carlo_control(num_episodes=num_episodes)

    print(f"\n[Training Q-Learning agent ({num_episodes} episodes) ...]")
    Q_ql, policy_ql, ep_rewards = q_learning()

    V_ql = q_to_V(Q_ql)
    print_value_function(V_ql, "State-Value Function  V(s) = max_a Q(s,a)  [Q-Learning]")
    print_policy(policy_ql, "Learned Policy (Q-Learning)")

    print_policy(policy_mc, "Reference: Learned Policy (Monte Carlo – Task 2)")
    print_policy(policy_vi, "Reference: Optimal Policy (Value Iteration – Task 1)")

    print("\n[Policy Comparisons]")
    compare_policies(policy_ql, policy_vi, label_a="Q-Learning", label_b="Optimal (Value Iteration)")
    compare_policies(policy_ql, policy_mc, label_a="Q-Learning", label_b="MC Control")
    compare_policies(policy_mc, policy_vi, label_a="MC Control", label_b="Optimal (Value Iteration)")

    chunk = 5_000
    print("\n[Q-Learning Convergence: Average Episode Reward]")
    for i in range(0, num_episodes, chunk):
        window = ep_rewards[i: i + chunk]
        avg = sum(window) / len(window)
        print(f"    Episodes {i+1:>6}–{i+chunk:<6}: avg reward = {avg:+.2f}")
    print()
