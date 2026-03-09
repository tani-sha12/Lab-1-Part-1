import pickle
import heapq
import math

# Load data (place .pkl files in the same folder as this script)
with open('G.pkl', 'rb') as f:
    G = pickle.load(f)
with open('Dist.pkl', 'rb') as f:
    Dist = pickle.load(f)
with open('Cost.pkl', 'rb') as f:
    Cost = pickle.load(f)

SOURCE = '1'
TARGET = '50'


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
    print(f"Shortest path: {path_str}.")
    print(f"Shortest distance: {total_dist}.")
    print(f"Total energy cost: {total_energy}.")


# Task 1 — Uniform-Cost Search (no energy constraint)
# Finds the true shortest-distance path from SOURCE to TARGET,
# ignoring energy costs.
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
            new_dist = d + Dist[u, v]
            if new_dist < best_dist.get(v, math.inf):
                best_dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    path = reconstruct_path(prev, source, target)
    if path is None:
        print("No path found.")
        return

    total_dist = best_dist[target]
    total_energy = sum(Cost[path[i], path[i + 1]] for i in range(len(path) - 1))
    return path, total_dist, total_energy


if __name__ == '__main__':
    print("=" * 60)
    print("Task 1: UCS (no energy constraint)")
    print("=" * 60)
    result = task1_ucs(G, Dist, SOURCE, TARGET)
    if result:
        path, dist, energy = result
        format_output(path, dist, energy)
