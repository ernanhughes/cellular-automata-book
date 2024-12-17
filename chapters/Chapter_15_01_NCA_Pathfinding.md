# Chapter: Pathfinding Neural Cellular Automata (NCA)

Pathfinding tasks are fundamental in robotics, game development, and artificial intelligence. Neural Cellular Automata (NCA) offers a unique approach to solving these problems by leveraging the self-organizing behavior of cellular automata, augmented with neural networks to learn complex dynamics. This chapter explores the concept of pathfinding using NCA, delves into the principles behind it, and provides Python implementations to demonstrate its power.

---

## **1. Introduction to Pathfinding with NCA**

Pathfinding involves finding the optimal route from a start point to a destination. Traditional algorithms like A* and Dijkstra are widely used for this purpose. However, NCA provides an alternative paradigm: cells in a grid represent the environment and collaboratively learn to construct paths through local interactions.

### **Why Use NCA for Pathfinding?**
1. **Decentralization**: Each cell independently updates based on its neighbors, making the approach scalable.
2. **Learning Capability**: Neural networks enable the automaton to learn optimal strategies for complex environments.
3. **Robustness**: Self-organizing dynamics adapt to changes in the environment, such as obstacles.

---

## **2. Building Blocks of Pathfinding NCA**

### **2.1 Cellular Representation**
- The grid represents the environment.
- Each cell has states indicating whether it is:
  - Start point
  - Target point
  - Obstacle
  - Part of the path

### **2.2 Neural Cellular Automata**
- **State Vector**: Each cell is represented by a state vector containing information about its current state.
- **Update Rule**: A neural network models the update rule, which determines the next state of the cell based on its current state and neighbors.

### **2.3 Training the NCA**
- Use supervised learning to train the neural network.
- Loss function: Encourage the NCA to correctly connect the start and end points while avoiding obstacles.

---

## **3. Python Implementation of Pathfinding NCA**

### **3.1 Environment Setup**
The environment is represented as a 2D grid.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define grid size
grid_size = 20

# Initialize grid
# 0: Empty, 1: Start, 2: Target, 3: Obstacle

def initialize_grid(size, obstacle_ratio=0.2):
    grid = np.zeros((size, size), dtype=int)
    
    # Place start and target points
    grid[0, 0] = 1  # Start
    grid[-1, -1] = 2  # Target

    # Add obstacles
    num_obstacles = int(size * size * obstacle_ratio)
    for _ in range(num_obstacles):
        x, y = np.random.randint(0, size, size=2)
        if grid[x, y] == 0:
            grid[x, y] = 3
    
    return grid

def display_grid(grid):
    plt.imshow(grid, cmap="viridis", interpolation="none")
    plt.colorbar()
    plt.show()

# Initialize and display the grid
grid = initialize_grid(grid_size)
display_grid(grid)
```

---

### **3.2 Neural Network for NCA**
Define a neural network to update the state of each cell based on its neighbors.

```python
import torch
import torch.nn as nn

class PathfindingNCA(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=16):
        super(PathfindingNCA, self).__init__()
        self.conv = nn.Conv2d(state_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, state_dim, kernel_size=1)
        )

    def forward(self, x):
        dx = self.conv(x)
        dx = self.fc(dx)
        return x + dx

# Initialize the NCA model
state_dim = 4  # 4 states: empty, start, target, obstacle
nca = PathfindingNCA(state_dim)
```

---

### **3.3 Training the NCA**
Define a loss function and train the NCA to construct paths.

```python
def loss_function(predicted_grid, target_grid):
    # Encourage correct path connections
    path_loss = torch.mean((predicted_grid[:, 3, :, :] - target_grid[:, 3, :, :]) ** 2)
    return path_loss

# Training loop
def train_nca(nca, grid, target_grid, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(nca.parameters(), lr=lr)
    
    # Convert grids to tensors
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target_grid, dtype=torch.float32).unsqueeze(0)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_grid = nca(grid_tensor)
        
        # Compute loss
        loss = loss_function(predicted_grid, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate target grid (ideal pathfinding solution)
target_grid = grid.copy()
# Assume a known solution for demonstration purposes
train_nca(nca, grid, target_grid)
```

---

### **3.4 Testing the NCA**
Visualize the results of the trained NCA on the grid.

```python
def test_nca(nca, grid, steps=20):
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    for step in range(steps):
        grid_tensor = nca(grid_tensor)
        
        # Visualize the current state
        current_grid = grid_tensor.squeeze(0).detach().numpy()
        plt.imshow(current_grid[3], cmap="viridis")
        plt.title(f"Step {step}")
        plt.show()

# Test the NCA on the grid
test_nca(nca, grid)
```

---

## **4. Summary**
Pathfinding using Neural Cellular Automata offers a novel and decentralized approach to solve routing problems. By training NCA to construct paths in complex environments, we unlock new possibilities in adaptive path planning, robotics, and AI-driven simulations.

In this chapter, we covered:
1. The principles of pathfinding NCA.
2. Python implementations for grid setup, NCA modeling, and training.
3. Testing and visualization of the results.

Further research can extend this approach to handle dynamic obstacles, larger environments, or multi-agent systems.

