# Section 3.2 — For Each Task, Explain Your Approach, Output, etc.

---

## Part 1 — Finding a Shortest Path with an Energy Budget

**Problem Setup:**
The NYC road network is modelled as a directed graph loaded from JSON files (`G.json`, `Dist.json`, `Cost.json`, `Coord.json`). Each edge carries two independent weights: a distance (metres) and an energy cost (Joules). The task is to find the shortest-distance path from node **1** to node **50**, subject to an energy budget of **287,932** units.

---

### Task 1: Shortest Path (No Energy Constraint)

**Algorithm:** Uniform-Cost Search (UCS / Dijkstra's algorithm)

**Approach:**
Task 1 ignores the energy dimension entirely and finds the single shortest-distance path from source to target. The implementation uses a min-heap priority queue ordered by accumulated distance. At each step the node with the smallest known distance is expanded; neighbours are relaxed if a shorter path is discovered. Expansion terminates as soon as the target node is popped from the heap (this is correct because all edge weights are non-negative, so the first time a node is popped its distance is already optimal).

Key implementation details:

- State: `(accumulated_distance, node)`
- A `best_dist` dictionary tracks the minimum distance seen for each node; stale heap entries (where the popped distance exceeds the recorded best) are discarded.
- The predecessor map `prev` is used to reconstruct the path after the search terminates.
- Energy cost of the resulting path is computed post-hoc for reporting.

**Output:**

```
Shortest path: 1->1363->1358->1357->1356->1276->1273->1277->1269->1267->1268->
1284->1283->1282->1255->1253->1260->1259->1249->1246->963->964->962->1002->952->
1000->998->994->995->996->987->988->979->980->969->977->989->990->991->2369->
2366->2340->2338->2339->2333->2334->2329->2029->2027->2019->2022->2000->1996->
1997->1993->1992->1989->1984->2001->1900->1875->1874->1965->1963->1964->1923->
1944->1945->1938->1937->1939->1935->1931->1934->1673->1675->1674->1837->1671->
1828->1825->1817->1815->1634->1814->1813->1632->1631->1742->1741->1740->1739->
1591->1689->1585->1584->1688->1579->1679->1677->104->5680->5418->5431->5425->
5424->5422->5413->5412->5411->66->5392->5391->5388->5291->5278->5289->5290->
5283->5284->5280->50.
Shortest distance: 148648.637.
Total energy cost: 294853.
```

**Observation:** The unconstrained shortest path has a total energy cost of **294,853**, which exceeds the budget of 287,932. This path cannot be used for Tasks 2 and 3 — a longer (in distance) but more energy-efficient route must be found.

---

### Task 2: Energy-Constrained Shortest Path (Uninformed Search)

**Algorithm:** Energy-Constrained Uniform-Cost Search

**Approach:**
Task 2 extends UCS with an energy dimension. Because energy is a secondary constraint (not the optimisation objective), the state must track both distance and energy accumulated so far. The priority queue is ordered by distance, but energy forms a hard feasibility constraint.

Key implementation details:

- State: `(accumulated_distance, accumulated_energy, node)`
- Any successor state whose energy would exceed `ENERGY_BUDGET = 287,932` is pruned immediately before being added to the queue.
- The `best` dictionary maps each node to its best known `(distance, energy)` pair. A new state for node `v` is only enqueued if it achieves a strictly smaller distance, or the same distance with strictly less energy, compared to the current best for `v`. This dominance pruning is crucial for tractability.
- A separate `visited` dictionary records the `(distance, energy)` at the time a node was expanded; re-expansions that are dominated by the visited state are skipped.

This is an **uninformed** search because it does not use any heuristic to guide exploration — nodes are expanded purely in order of their accumulated distance from the source.

**Output:**

```
Shortest path: 1->1363->1358->1357->1356->1276->1273->1277->1269->1267->1268->
1284->1283->1282->1255->1253->1260->1259->1249->1246->963->964->962->1002->952->
1000->998->994->995->996->987->988->979->980->969->977->989->990->991->2369->
2366->2340->2338->2339->2333->2334->2329->2029->2027->2019->2022->2000->1996->
1997->1993->1992->1989->1984->2001->1900->1875->1874->1965->1963->1964->1923->
1944->1945->1938->1937->1939->1935->1931->1934->1673->1675->1674->1837->1671->
1828->1825->1817->1815->1634->1814->1813->1632->1631->1742->1741->1740->1739->
1591->1689->1585->1584->1688->1579->1679->1677->104->5680->5418->5431->5425->
5429->5426->5428->5434->5435->5433->5436->5398->5404->5402->5396->5395->5292->
5282->5283->5284->5280->50.
Shortest distance: 150784.607.
Total energy cost: 287931.
```

**Observation:** The energy-constrained path is ~2,136 metres longer than the unconstrained path but uses energy **287,931 ≤ 287,932**, just within budget. The path diverges from Task 1's path near the end (node 5425 onward), taking a different route to the destination.

---

### Task 3: Energy-Constrained Shortest Path (Informed Search — A\*)

**Algorithm:** A\* Search with Euclidean heuristic and energy pruning

**Approach:**
Task 3 solves the same energy-constrained problem as Task 2, but with an **informed** heuristic to accelerate the search. The heuristic `h(n)` is the straight-line (Euclidean) distance from node `n` to the target, computed from geographic coordinates in `Coord.json`.

The total priority of a node is `f = g + h`, where `g` is the actual distance accumulated so far and `h` is the Euclidean estimate of the remaining distance to the target.

**Heuristic admissibility:**
A heuristic is admissible if it never overestimates the true remaining cost. Since road distances can only be ≥ straight-line distances (roads must travel at least as far as the crow flies), `h(n) = euclidean_distance(n, target)` never overestimates the remaining road distance, making it **admissible**. An admissible heuristic guarantees that A\* finds the optimal solution.

**Heuristic consistency (monotonicity):**
A heuristic is consistent if `h(n) ≤ cost(n, n') + h(n')` for every edge `(n, n')`. Euclidean distance satisfies the triangle inequality: the straight-line distance from `n` to `target` is at most the edge distance `cost(n, n')` plus the straight-line distance from `n'` to `target`. Consistency implies admissibility and also means A\* never needs to re-open a closed node, making it more efficient.

Key implementation details:

- State in heap: `(f_score, g_score, energy, node)`
- The `best` dictionary stores the best `(g, energy)` seen for each node; outdated heap entries are discarded when popped.
- Energy pruning is identical to Task 2: successors exceeding the budget are discarded before insertion.
- The heuristic guides the search toward the target more directly, expanding fewer nodes than UCS in Task 2.

**Output:**

```
Shortest path: 1->1363->1358->1357->1356->1276->1273->1277->1269->1267->1268->
1284->1283->1282->1255->1253->1260->1259->1249->1246->963->964->962->1002->952->
1000->998->994->995->996->987->988->979->980->969->977->989->990->991->2369->
2366->2340->2338->2339->2333->2334->2329->2029->2027->2019->2022->2000->1996->
1997->1993->1992->1989->1984->2001->1900->1875->1874->1965->1963->1964->1923->
1944->1945->1938->1937->1939->1935->1931->1934->1673->1675->1674->1837->1671->
1828->1825->1817->1815->1634->1814->1813->1632->1631->1742->1741->1740->1739->
1591->1689->1585->1584->1688->1579->1679->1677->104->5680->5418->5431->5425->
5429->5426->5428->5434->5435->5433->5436->5398->5404->5402->5396->5395->5292->
5282->5283->5284->5280->50.
Shortest distance: 150784.607.
Total energy cost: 287931.
```

**Observation:** A\* produces the **same optimal path** as energy-constrained UCS (Task 2) — both find the shortest feasible path of distance 150,784.607 with energy 287,931. A\* achieves this by directing the search toward the goal using the heuristic, typically expanding far fewer nodes than UCS while still guaranteeing optimality.

---

## Part 2 — Solving MDP and RL Problems Using a Grid World

**Environment:**
A 5×5 stochastic grid world:

- **Start:** (0, 0) — bottom-left corner
- **Goal:** (4, 4) — top-right corner (terminal state)
- **Roadblocks:** (2, 1) and (2, 3) — impassable cells
- **Actions:** Up (^), Down (v), Left (<), Right (>)
- **Stochastic transitions:** When action `a` is chosen, the agent moves in the intended direction with probability 0.8, or in a perpendicular direction with probability 0.1 each. If the resulting cell is out-of-bounds or a roadblock, the agent stays in place.
- **Rewards:** −1 per step; +10 on reaching the goal
- **Discount factor:** γ = 0.9

---

### Task 1: Value Iteration and Policy Iteration

**Algorithms:** (a) Value Iteration, (b) Policy Iteration — both classical MDP planning methods that assume full knowledge of the transition model.

#### (a) Value Iteration

**Approach:**
Value Iteration applies the Bellman optimality operator repeatedly until the value function converges:

```
V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]
```

Starting from `V(s) = 0` for all states, each sweep updates every non-terminal state. Convergence is declared when the maximum change across all states `Δ < θ = 1e-9`. After convergence, the optimal policy is extracted greedily: for each state, choose the action that maximises the Bellman backup value.

**Output:**

```
Value Iteration converged after 37 sweeps (Δ = 4.62e-10)

Optimal Value Function (Value Iteration):
         x=0    x=1    x=2    x=3    x=4
  y=4 | +2.71  +4.65  +6.93  +9.28  [GOAL]
  y=3 | +1.31  +2.71   BLK   +7.16  +9.28
  y=2 | +0.16  +1.56  +3.22  +5.05  +6.74
  y=1 | -0.97  +0.04   BLK   +3.35  +4.57
  y=0 | -1.90  -0.90  +0.25  +1.68  +2.68

Optimal Policy (Value Iteration):
         x=0   x=1   x=2   x=3   x=4
  y=4 |   >     >     >     >     G
  y=3 |   ^     ^     B     ^     ^
  y=2 |   >     >     >     ^     ^
  y=1 |   ^     ^     B     ^     ^
  y=0 |   >     >     >     ^     ^
```

#### (b) Policy Iteration

**Approach:**
Policy Iteration alternates between two steps:

1. **Policy Evaluation:** Given the current policy π, compute V^π by iterating the Bellman expectation equation to convergence (θ = 1e-9):
   ```
   V^π_{k+1}(s) = Σ_{s'} P(s'|s, π(s)) [R(s,π(s),s') + γ V^π_k(s')]
   ```
2. **Policy Improvement:** For each state, update the policy greedily with respect to the current V^π. If no state's action changes, the policy is optimal and the algorithm terminates.

The algorithm starts with an arbitrary policy (always Up) and is guaranteed to converge to the optimal policy in a finite number of iterations.

**Output:**

```
Policy Iteration converged after 5 evaluation/improvement cycles

Optimal Value Function (Policy Iteration):
         x=0    x=1    x=2    x=3    x=4
  y=4 | +2.71  +4.65  +6.93  +9.28  [GOAL]
  y=3 | +1.31  +2.71   BLK   +7.16  +9.28
  y=2 | +0.16  +1.56  +3.22  +5.05  +6.74
  y=1 | -0.97  +0.04   BLK   +3.35  +4.57
  y=0 | -1.90  -0.90  +0.25  +1.68  +2.68

Optimal Policy (Policy Iteration):
         x=0   x=1   x=2   x=3   x=4
  y=4 |   >     >     >     >     G
  y=3 |   ^     ^     B     ^     ^
  y=2 |   >     >     >     ^     ^
  y=1 |   ^     ^     B     ^     ^
  y=0 |   >     >     >     ^     ^
```

#### Comparison: VI vs PI

```
✓ Policies from VI and PI are IDENTICAL.
Max |V_VI(s) − V_PI(s)| across all states: 0.00e+00
```

**Analysis:**
Both algorithms converge to the exact same optimal value function and policy, confirming correctness. VI required 37 full sweeps over all states, while PI converged in only 5 policy evaluation/improvement cycles. In general, Policy Iteration tends to require fewer iterations at the cost of a more expensive inner loop (full policy evaluation per cycle), whereas Value Iteration is simpler per sweep but may need more sweeps. For this small grid world, both are fast.

The optimal policy makes intuitive sense: from most states, the agent moves right and up toward the goal. The roadblocks at (2,1) and (2,3) force the agent to navigate around them — states to the left of the roadblocks move up or right depending on which route is shorter.

---

### Task 2: Monte Carlo Control

**Algorithm:** First-Visit Monte Carlo Control with ε-greedy exploration

**Approach:**
Unlike Task 1, the agent does **not** know the transition model. It must learn by interacting with the environment through complete episodes. The algorithm:

1. Initialises Q(s, a) = 0 for all state-action pairs.
2. For each episode:
   - Generate a complete trajectory from START to GOAL using the current ε-greedy policy (ε = 0.1): with probability 0.1, choose a random action; otherwise, choose the action with the highest Q-value (ties broken randomly).
   - Compute discounted returns G_t backwards from the end of the episode.
   - For each (state, action) pair encountered **for the first time** in the episode, update Q(s, a) as the running average of all observed returns.
3. After 50,000 episodes, extract the greedy policy from Q.

The **first-visit** variant only uses the first occurrence of each (s, a) pair per episode, providing unbiased return estimates under the behaviour policy.

**Hyperparameters:** ε = 0.1, γ = 0.9, 50,000 episodes, random seed = 42.

**Output:**

```
State-Value Function V(s) = max_a Q(s,a) [Monte Carlo]:
         x=0    x=1    x=2    x=3    x=4
  y=4 | -2.15  +2.49  +7.01  +9.48  [GOAL]
  y=3 | -1.97  -0.74   BLK   +7.21  +9.94
  y=2 | -0.72  +1.01  +2.73  +5.02  +7.23
  y=1 | -1.96  -0.77   BLK   +3.05  +4.58
  y=0 | -3.19  -2.61  -2.69  +0.59  +2.47

Learned Policy (Monte Carlo):
         x=0   x=1   x=2   x=3   x=4
  y=4 |   ^     ^     >     >     G
  y=3 |   v     v     B     >     ^
  y=2 |   >     >     >     >     ^
  y=1 |   ^     ^     B     ^     ^
  y=0 |   ^     ^     >     v     ^

Policy Comparison (MC vs Optimal / Value Iteration):
  Agreement: 13/22 states (59.1%)
  Mismatches at: (0,0), (0,3), (0,4), (1,0), (1,3), (1,4), (3,0), (3,2), (3,3)
```

**Analysis:**
Monte Carlo Control achieves 59.1% agreement with the optimal policy after 50,000 episodes. The mismatches are concentrated in states that are relatively far from the goal and rarely visited during training — particularly the bottom-left corner and cells adjacent to roadblocks. The ε-greedy policy ensures exploration, but rare states accumulate fewer return samples, leaving Q-value estimates noisier. The MC value function is also noisier than the exact values computed in Task 1 (e.g., state (0,4) shows −2.15 vs. +2.71 under VI), reflecting this estimation variance. With more episodes or decaying ε, the policy would converge closer to optimal.

---

### Task 3: Q-Learning

**Algorithm:** Tabular Q-Learning (off-policy TD control)

**Approach:**
Q-Learning is an **off-policy** temporal-difference algorithm that updates Q-values after every single step (no need to wait for episode completion). The update rule is:

```
Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') − Q(s, a)]
```

The key distinction from Monte Carlo is that the update target uses `max_{a'} Q(s', a')` — the value of the best action from the next state under the **greedy policy**, regardless of which action was actually taken. This makes Q-learning off-policy: it learns the optimal Q-function even while following an exploratory ε-greedy behaviour policy.

**Hyperparameters:** ε = 0.1, α = 0.1, γ = 0.9, 50,000 episodes, max 500 steps per episode, random seed = 42.

**Output:**

```
State-Value Function V(s) = max_a Q(s,a) [Q-Learning]:
         x=0    x=1    x=2    x=3    x=4
  y=4 | +3.27  +5.04  +7.17  +9.54  [GOAL]
  y=3 | +1.77  +2.85   BLK   +6.93  +9.20
  y=2 | +0.64  +1.94  +3.41  +5.06  +5.91
  y=1 | -0.76  +0.38   BLK   +3.22  +4.62
  y=0 | -1.83  -1.06  +0.03  +1.48  +3.12

Learned Policy (Q-Learning):
         x=0   x=1   x=2   x=3   x=4
  y=4 |   >     >     >     >     G
  y=3 |   ^     ^     B     ^     ^
  y=2 |   >     >     >     >     ^
  y=1 |   ^     ^     B     ^     ^
  y=0 |   ^     >     >     >     ^
```

**Policy Comparisons:**

```
Q-Learning vs Optimal (Value Iteration):
  Agreement: 19/22 states (86.4%)
  Mismatches: (0,0) QL=U Opt=R, (3,0) QL=R Opt=U, (3,2) QL=R Opt=U

Q-Learning vs MC Control:
  Agreement: 15/22 states (68.2%)

MC Control vs Optimal (Value Iteration):
  Agreement: 13/22 states (59.1%)
```

**Convergence:**

```
Episodes      1–5000  : avg reward = -0.19
Episodes   5001–10000 : avg reward = +0.09
Episodes  10001–15000 : avg reward = +0.09
Episodes  15001–20000 : avg reward = +0.08
Episodes  20001–25000 : avg reward = +0.05
Episodes  25001–30000 : avg reward = +0.06
Episodes  30001–35000 : avg reward = +0.09
Episodes  35001–40000 : avg reward = +0.12
Episodes  40001–45000 : avg reward = +0.04
Episodes  45001–50000 : avg reward = +0.04
```

**Analysis:**
Q-Learning substantially outperforms Monte Carlo Control, achieving **86.4% policy agreement** with the optimal policy (19/22 states) compared to MC's 59.1% (13/22 states). The Q-Learning value function is also much closer to the true values from Task 1 — e.g., state (0,4) shows +3.27 (vs +2.71 under VI), a much smaller discrepancy than MC's −2.15.

The convergence curve shows a sharp improvement from the initial −0.19 average reward (early exploration, random policy) to approximately +0.05–0.12 in subsequent windows. The relatively stable positive average rewards from episode 5,000 onward indicate the agent has learned a reasonably effective policy early in training.

Q-Learning's advantage over MC here stems from its **online, step-level updates**: every step contributes to learning, so all states are updated more uniformly throughout training — including rare states far from the goal. Monte Carlo must wait for episode completion to back-propagate returns, and episodes that take many steps give high-variance return estimates. Q-Learning's bootstrapped targets reduce this variance at the cost of some bias (from using current Q-estimates in the target), but in practice converges faster and more reliably on this problem.

The three remaining mismatches between Q-Learning and optimal — at states (0,0), (3,0), and (3,2) — are states where multiple actions have very similar Q-values, and the stochastic transitions make the value differences between actions small relative to estimation noise.

**Summary comparison across all methods:**

| Method              | Agreement with Optimal | Notes                                             |
| ------------------- | ---------------------- | ------------------------------------------------- |
| Value Iteration     | 100% (exact)           | Requires full model knowledge                     |
| Policy Iteration    | 100% (exact)           | Requires full model knowledge; faster convergence |
| Monte Carlo Control | 59.1% (13/22 states)   | Model-free; high variance, slower convergence     |
| Q-Learning          | 86.4% (19/22 states)   | Model-free; lower variance, faster convergence    |
