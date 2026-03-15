import json
import heapq
import math

# Load data (place .json files in the same folder as this script)
with open('G.json') as f:
    G = json.load(f)
with open('Dist.json') as f:
    Dist = json.load(f)
with open('Cost.json') as f:
    Cost = json.load(f)

SOURCE = '1'
TARGET = '50'
ENERGY_BUDGET = 287932


def reconstruct_path(prev, source, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    if path[0] != source:
        return None
    return path


def format_output(path, total_dist, total_energy):
    path_str = '->'.join(path)
    print(f"Shortest path: {path_str}")
    print(f"Shortest distance: {total_dist}")
    print(f"Total energy cost: {total_energy}")


# Task 2 — Uniform-Cost Search (energy-constrained)
# Finds the shortest-distance path whose total energy <= ENERGY_BUDGET.
# State: (accumulated_dist, accumulated_energy, node)
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

            if v in best:
                bd, be = best[v]
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


if __name__ == '__main__':
    print("=" * 60)
    print(f"Task 2: UCS (energy budget = {ENERGY_BUDGET})")
    print("=" * 60)
    result = task2_ucs(G, Dist, Cost, SOURCE, TARGET, ENERGY_BUDGET)
    if result:
        path, dist, energy = result
        format_output(path, dist, energy)
