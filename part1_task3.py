import json
import heapq
import math

# Load data (place .json files in the same folder as this script)
with open('G.json') as f:
    G = json.load(f)
with open('Coord.json') as f:
    Coord = json.load(f)
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


def euclidean_heuristic(node, target, Coord):
    x1, y1 = Coord[node]
    x2, y2 = Coord[target]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Task 3 — A* Search (energy-constrained)
# Uses Euclidean straight-line distance to TARGET as an admissible heuristic.
# State: (f = g + h, g = dist_so_far, energy, node)
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

        if u in best:
            bg, be = best[u]
            if g > bg or (g == bg and e > be):
                continue

        for v in G[u]:
            new_g = g + Dist[f"{u},{v}"]
            new_e = e + Cost[f"{u},{v}"]

            if new_e > budget:
                continue  # prune: energy constraint violated

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


if __name__ == '__main__':
    print("=" * 60)
    print(f"Task 3: A* (energy budget = {ENERGY_BUDGET})")
    print("=" * 60)
    result = task3_astar(G, Dist, Cost, Coord, SOURCE, TARGET, ENERGY_BUDGET)
    if result:
        path, dist, energy = result
        format_output(path, dist, energy)
