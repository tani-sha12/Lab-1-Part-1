import json
import heapq
import math

# ──────────────────────────────────────────────
# Load instance data
# ──────────────────────────────────────────────
with open('G.json') as f:
    G = json.load(f)

with open('Coord.json') as f:
    Coord = json.load(f)

with open('Dist.json') as f:
    Dist = json.load(f)

with open('Cost.json') as f:
    Cost = json.load(f)

# ──────────────────────────────────────────────
# Problem parameters
# ──────────────────────────────────────────────
SOURCE = '1'
TARGET = '50'
ENERGY_BUDGET = 287932


# ──────────────────────────────────────────────
# Helper: reconstruct path from predecessor map
# ──────────────────────────────────────────────
def reconstruct_path(prev, source, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    if path[0] != source:
        return None  # no path found
    return path


def format_output(path, total_dist, total_energy):
    path_str = '->'.join(path)
    print(f"Shortest path: {path_str}.")
    print(f"Shortest distance: {total_dist}.")
    print(f"Total energy cost: {total_energy}.")


# ══════════════════════════════════════════════
# Task 1 — Uniform-Cost Search (no energy constraint)
# Finds the true shortest-distance path from
# SOURCE to TARGET, ignoring energy costs.
# ══════════════════════════════════════════════
def task1_ucs(G, Dist, source, target):
    # Priority queue: (distance, node)
    pq = [(0, source)]
    best_dist = {source: 0}
    prev = {source: None}

    while pq:
        d, u = heapq.heappop(pq)

        if u == target:
            break

        if d > best_dist.get(u, math.inf):
            continue

        for v in G[u]:
            new_dist = d + Dist[f"{u},{v}"]
            if new_dist < best_dist.get(v, math.inf):
                best_dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    path = reconstruct_path(prev, source, target)
    if path is None:
        print("No path found.")
        return

    total_dist = best_dist[target]
    total_energy = sum(Cost[f"{path[i]},{path[i+1]}"] for i in range(len(path) - 1))
    return path, total_dist, total_energy


# ══════════════════════════════════════════════
# Task 2 — Uniform Cost Search (energy-constrained)
# An uninformed search that finds the shortest-
# distance path whose total energy ≤ ENERGY_BUDGET.
# State: (accumulated_dist, accumulated_energy, node)
# ══════════════════════════════════════════════
def task2_ucs(G, Dist, Cost, source, target, budget):
    # Priority queue: (distance, energy, node)
    pq = [(0, 0, source)]
    visited = {}  # node -> (best_dist, best_energy) at time of expansion
    prev = {source: None}
    best = {source: (0, 0)}  # node -> (dist, energy)

    while pq:
        d, e, u = heapq.heappop(pq)

        if u == target:
            break

        # Skip if we already found a better state for this node
        if u in visited:
            vd, ve = visited[u]
            if d >= vd and e >= ve:
                continue

        visited[u] = (d, e)

        for v in G[u]:
            new_dist = d + Dist[f"{u},{v}"]
            new_energy = e + Cost[f"{u},{v}"]

            if new_energy > budget:
                continue  # prune: energy constraint violated

            # Only enqueue if this is a better (dist, energy) for v
            if v in best:
                bd, be = best[v]
                # Accept if shorter distance, or same dist with less energy
                if new_dist > bd or (new_dist == bd and new_energy >= be):
                    continue

            best[v] = (new_dist, new_energy)
            prev[v] = u
            heapq.heappush(pq, (new_dist, new_energy, v))

    path = reconstruct_path(prev, source, target)
    if path is None:
        print("No feasible path found within energy budget.")
        return

    total_dist = sum(Dist[f"{path[i]},{path[i+1]}"] for i in range(len(path) - 1))
    total_energy = sum(Cost[f"{path[i]},{path[i+1]}"] for i in range(len(path) - 1))
    return path, total_dist, total_energy


# ══════════════════════════════════════════════
# Task 3 — A* Search (energy-constrained)
# Uses Euclidean straight-line distance to TARGET
# as an admissible, consistent heuristic h(n).
# State: (f = g + h, g = dist_so_far, energy, node)
# ══════════════════════════════════════════════
def euclidean_heuristic(node, target, Coord):
    x1, y1 = Coord[node]
    x2, y2 = Coord[target]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def task3_astar(G, Dist, Cost, Coord, source, target, budget):
    h_source = euclidean_heuristic(source, target, Coord)

    # Priority queue: (f_score, g_score, energy, node)
    pq = [(h_source, 0, 0, source)]
    best = {source: (0, 0)}  # node -> (best_dist, best_energy)
    prev = {source: None}

    while pq:
        f, g, e, u = heapq.heappop(pq)

        if u == target:
            break

        # Skip outdated entries
        if u in best:
            bg, be = best[u]
            if g > bg or (g == bg and e > be):
                continue

        for v in G[u]:
            new_g = g + Dist[f"{u},{v}"]
            new_e = e + Cost[f"{u},{v}"]

            if new_e > budget:
                continue  # prune: energy constraint violated

            # Check if this state is better than what we've seen for v
            if v in best:
                bg, be = best[v]
                if new_g > bg or (new_g == bg and new_e >= be):
                    continue

            best[v] = (new_g, new_e)
            prev[v] = u
            h = euclidean_heuristic(v, target, Coord)
            heapq.heappush(pq, (new_g + h, new_g, new_e, v))

    path = reconstruct_path(prev, source, target)
    if path is None:
        print("No feasible path found within energy budget.")
        return

    total_dist = sum(Dist[f"{path[i]},{path[i+1]}"] for i in range(len(path) - 1))
    total_energy = sum(Cost[f"{path[i]},{path[i+1]}"] for i in range(len(path) - 1))
    return path, total_dist, total_energy


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Task 1: UCS (no energy constraint)")
    print("=" * 60)
    result = task1_ucs(G, Dist, SOURCE, TARGET)
    if result:
        path, dist, energy = result
        format_output(path, dist, energy)

    print()
    print("=" * 60)
    print(f"Task 2: UCS (energy budget = {ENERGY_BUDGET})")
    print("=" * 60)
    result = task2_ucs(G, Dist, Cost, SOURCE, TARGET, ENERGY_BUDGET)
    if result:
        path, dist, energy = result
        format_output(path, dist, energy)

    print()
    print("=" * 60)
    print(f"Task 3: A* (energy budget = {ENERGY_BUDGET})")
    print("=" * 60)
    result = task3_astar(G, Dist, Cost, Coord, SOURCE, TARGET, ENERGY_BUDGET)
    if result:
        path, dist, energy = result
        format_output(path, dist, energy)
