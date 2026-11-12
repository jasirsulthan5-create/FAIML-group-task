"""Module 2 - Big Data Process Mapping (Google Maps case study).
Dijkstra's shortest path on the toy road graph of Figure 2.2.
Edge weights are predicted travel times in minutes (illustrative)."""
import heapq

GRAPH = {
    "A": {"B": 4, "C": 5},
    "B": {"A": 4, "C": 2, "D": 11},   # B-D is congested (expensive)
    "C": {"A": 5, "B": 2, "E": 7},
    "D": {"B": 11, "E": 3, "F": 5, "G": 9},
    "E": {"C": 7, "D": 3, "F": 4},
    "F": {"D": 5, "E": 4, "G": 3},
    "G": {"D": 9, "F": 3},
}


def dijkstra(graph, start, goal):
    pq = [(0, start, [start])]
    seen = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            return cost, path
        for nxt, w in graph[node].items():
            if nxt not in seen:
                heapq.heappush(pq, (cost + w, nxt, path + [nxt]))
    return float("inf"), []


if __name__ == "__main__":
    cost, path = dijkstra(GRAPH, "A", "G")
    print("Route:", "-".join(path), f"({cost} minutes)")
