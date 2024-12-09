Hopcroft's algorithm is a graph theory algorithm used to find the maximum matching in a bipartite graph in \(O(E \sqrt{V})\) time. Below is a Python implementation of Hopcroft's algorithm.

---

### Python Implementation of Hopcroft's Algorithm

```python
from collections import deque, defaultdict

class BipartiteGraph:
    def __init__(self, left_nodes, right_nodes):
        """
        Initializes the bipartite graph.

        Args:
            left_nodes (list): Nodes on the left set.
            right_nodes (list): Nodes on the right set.
        """
        self.left_nodes = left_nodes
        self.right_nodes = right_nodes
        self.edges = defaultdict(list)  # Adjacency list
        self.pair_u = {u: None for u in left_nodes}  # Matchings for left nodes
        self.pair_v = {v: None for v in right_nodes}  # Matchings for right nodes
        self.dist = {}

    def add_edge(self, u, v):
        """Adds an edge between u (left node) and v (right node)."""
        if u in self.left_nodes and v in self.right_nodes:
            self.edges[u].append(v)
        else:
            raise ValueError("Nodes must belong to the appropriate sets.")

    def bfs(self):
        """Breadth-first search to find all augmenting paths."""
        queue = deque()
        for u in self.left_nodes:
            if self.pair_u[u] is None:  # Unmatched node
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = float('inf')

        self.dist[None] = float('inf')

        while queue:
            u = queue.popleft()
            if self.dist[u] < self.dist[None]:
                for v in self.edges[u]:
                    next_u = self.pair_v[v]
                    if self.dist[next_u] == float('inf'):
                        self.dist[next_u] = self.dist[u] + 1
                        queue.append(next_u)

        return self.dist[None] != float('inf')

    def dfs(self, u):
        """Depth-first search to augment paths."""
        if u is not None:
            for v in self.edges[u]:
                next_u = self.pair_v[v]
                if self.dist[next_u] == self.dist[u] + 1:
                    if self.dfs(next_u):
                        self.pair_v[v] = u
                        self.pair_u[u] = v
                        return True
            self.dist[u] = float('inf')
            return False
        return True

    def hopcroft_karp(self):
        """Finds the maximum matching using Hopcroft's algorithm."""
        matching = 0
        while self.bfs():
            for u in self.left_nodes:
                if self.pair_u[u] is None:
                    if self.dfs(u):
                        matching += 1
        return matching


# Example Usage
left_nodes = ['A', 'B', 'C']
right_nodes = [1, 2, 3]

graph = BipartiteGraph(left_nodes, right_nodes)

# Add edges
graph.add_edge('A', 1)
graph.add_edge('A', 2)
graph.add_edge('B', 2)
graph.add_edge('C', 3)

# Find the maximum matching
max_matching = graph.hopcroft_karp()
print(f"Maximum Matching Size: {max_matching}")

# Print the matchings
for u in graph.left_nodes:
    if graph.pair_u[u]:
        print(f"{u} -> {graph.pair_u[u]}")
```

---

### Explanation

1. **Initialization**:
   - The bipartite graph is represented with two sets of nodes (`left_nodes` and `right_nodes`).
   - `pair_u` and `pair_v` store the matching for each node in the left and right sets, respectively.

2. **Breadth-First Search (BFS)**:
   - Finds all shortest augmenting paths in the graph.
   - Tracks distances using a `dist` dictionary.

3. **Depth-First Search (DFS)**:
   - Augments paths identified by BFS.
   - Updates matchings for nodes along the path.

4. **Iterative Process**:
   - Alternates between BFS and DFS to iteratively increase the matching size.
   - Stops when no augmenting paths are found.

5. **Output**:
   - Returns the size of the maximum matching and the matchings themselves.

---

### Example Output

Given the input graph:

```
Left Nodes: ['A', 'B', 'C']
Right Nodes: [1, 2, 3]
Edges: A-1, A-2, B-2, C-3
```

The algorithm outputs:
```
Maximum Matching Size: 3
A -> 1
B -> 2
C -> 3
```

---

### Complexity
- **Time Complexity**: \(O(E \sqrt{V})\), where \(E\) is the number of edges, and \(V\) is the number of nodes.
- **Space Complexity**: \(O(E + V)\).

---

### Further Reading
1. **Hopcroft-Karp Algorithm**:
   - [Wikipedia: Hopcroft–Karp Algorithm](https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm)
2. **Graph Matching**:
   - [Introduction to Graph Matching](https://www.geeksforgeeks.org/hopcroft-karp-algorithm-for-maximum-matching/)
3. **Visualization Tools**:
   - [NetworkX Documentation](https://networkx.org/documentation/stable/)

This implementation can be extended for additional functionality, such as handling weighted edges or non-standard graph structures.