## Cave generation using CA

```Python
shape = (42,42)
WALL = 0
FLOOR = 1
fill_prob = 0.4

new_map = np.ones(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        choice = random.uniform(0, 1)
        new_map[i][j] = WALL if choice < fill_prob else FLOOR

# run for 6 generations
generations = 6
for generation in range(generations):
    for i in range(shape[0]):
        for j in range(shape[1]):
            # get the number of walls 1 away from each index
            # get the number of walls 2 away from each index
            submap = new_map[max(i-1, 0):min(i+2, new_map.shape[0]),max(j-1, 0):min(j+2, new_map.shape[1])]
            wallcount_1away = len(np.where(submap.flatten() == WALL)[0])
            submap = new_map[max(i-2, 0):min(i+3, new_map.shape[0]),max(j-2, 0):min(j+3, new_map.shape[1])]
            wallcount_2away = len(np.where(submap.flatten() == WALL)[0])
            # this consolidates walls
            # for first five generations build a scaffolding of walls
            if generation < 5:
                # if looking 1 away in all directions you see 5 or more walls
                # consolidate this point into a wall, if that doesnt happpen
                # and if looking 2 away in all directions you see less than
                # 7 walls, add a wall, this consolidates and adds walls
                if wallcount_1away >= 5 or wallcount_2away <= 7:
                    new_map[i][j] = WALL
                else:
                    new_map[i][j] = FLOOR
            # this consolidates open space, fills in standalone walls,
            # after generation 5 consolidate walls and increase walking space
            # if there are more than 5 walls nearby make that point a wall,
            # otherwise add a floor
            else:
                # if looking 1 away in all direction you see 5 walls
                # consolidate this point into a wall,
                if wallcount_1away >= 5:
                    new_map[i][j] = WALL
                else:
                    new_map[i][j] = FLOOR

display_cave(new_map)

```


# Chapter X: Generating Procedural Caves with Cellular Automata in Python

## Introduction

Procedural generation of caves is a popular application of cellular automata in game development. By simulating the natural processes of erosion and structure formation, cellular automata can produce organic and realistic cave systems. In this chapter, we demonstrate how to generate caves using cellular automata within a Jupyter notebook.

---

## Step 1: Setting Up the Cellular Automata Framework

We begin by creating a simple cellular automata implementation for generating cave-like structures.

```python
import numpy as np
import matplotlib.pyplot as plt
import random

def initialize_grid(size, fill_probability):
    """Initialize a grid with random 0s and 1s based on fill probability."""
    return np.random.choice([0, 1], size=size, p=[1-fill_probability, fill_probability])

def display_grid(grid):
    """Visualize the grid using matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary')
    plt.axis('off')
    plt.show()
```

### Example: Initializing and Displaying a Random Grid

```python
size = (64, 64)  # Grid size
fill_probability = 0.45  # Probability of a cell being a wall (1)

# Initialize and display the grid
grid = initialize_grid(size, fill_probability)
display_grid(grid)
```

---

## Step 2: Implementing Cellular Automata Rules for Cave Generation

The rules for cave generation are based on neighbor counts. If a cell has too few neighboring walls, it becomes empty (0). Otherwise, it remains a wall (1).

```python
def apply_ca_rules(grid, birth_limit, survival_limit):
    """Apply cellular automata rules to the grid."""
    new_grid = grid.copy()
    rows, cols = grid.shape
    
    for x in range(rows):
        for y in range(cols):
            # Count the number of walls in the 8-neighborhood
            neighbors = grid[max(0, x-1):min(rows, x+2), max(0, y-1):min(cols, y+2)].sum() - grid[x, y]
            
            if grid[x, y] == 1:
                # Survival rule
                new_grid[x, y] = 1 if neighbors >= survival_limit else 0
            else:
                # Birth rule
                new_grid[x, y] = 1 if neighbors >= birth_limit else 0
    
    return new_grid
```

### Example: Applying CA Rules to Generate Caves

```python
birth_limit = 4  # Minimum neighbors to create a wall
survival_limit = 3  # Minimum neighbors to keep a wall

# Apply CA rules over multiple iterations
iterations = 5
for _ in range(iterations):
    grid = apply_ca_rules(grid, birth_limit, survival_limit)

display_grid(grid)
```

---

## Step 3: Adding Boundaries and Refining the Cave

To make the cave suitable for a game, we ensure the edges are walls and optionally refine the structure further.

```python
def add_boundaries(grid):
    """Add solid boundaries around the grid."""
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid

# Add boundaries to the cave
grid = add_boundaries(grid)
display_grid(grid)
```

---

## Step 4: Exporting the Cave Map

To use the generated cave in a game, we can export it as a text file or a JSON object.

```python
def save_to_file(grid, filename):
    """Save the grid to a text file."""
    np.savetxt(filename, grid, fmt='%d')

# Save the cave grid to a file
save_to_file(grid, "cave_map.txt")
print("Cave map saved to cave_map.txt")
```

---

## Step 5: Bringing It All Together

Let’s combine all the steps into a single function that generates and saves a cave.

```python
def generate_cave(size, fill_probability, birth_limit, survival_limit, iterations, filename):
    """Generate a cave and save it to a file."""
    grid = initialize_grid(size, fill_probability)
    for _ in range(iterations):
        grid = apply_ca_rules(grid, birth_limit, survival_limit)
    grid = add_boundaries(grid)
    save_to_file(grid, filename)
    display_grid(grid)

# Generate and save a cave
generate_cave(
    size=(64, 64),
    fill_probability=0.45,
    birth_limit=4,
    survival_limit=3,
    iterations=5,
    filename="cave_map.txt"
)
```

---

## Conclusion

Using cellular automata for cave generation is a powerful way to create organic and dynamic game environments. This chapter demonstrated a complete workflow, from initializing a grid to exporting a usable cave map for games. Experiment with different parameters to tailor the caves to your game’s needs!




Certainly, here's a Jupyter Notebook chapter demonstrating how to generate caves using cellular automata in Python:

**Chapter 1: Cave Generation with Cellular Automata**

**1.1 Introduction**

Cellular automata are a powerful tool for generating complex structures with simple rules. In this chapter, we will explore how to utilize a specific type of cellular automaton, the **Game of Life**, to create realistic-looking cave systems. 

**1.2 The Game of Life**

The Game of Life is a classic cellular automaton devised by John Conway. It operates on a grid of cells, each of which can be in one of two states: alive or dead. The rules of the Game of Life determine the next state of each cell based on the number of living neighbors it has:

* **Underpopulation:** If a cell has fewer than two living neighbors, it dies (becomes dead).
* **Overpopulation:** If a cell has more than three living neighbors, it dies (becomes dead).
* **Stasis or Survival:** If a cell has two or three living neighbors, it survives (remains alive).
* **Reproduction:** If a dead cell has exactly three living neighbors, it becomes alive.

**1.3 Cave Generation Algorithm**

To generate caves, we can adapt the Game of Life rules. Instead of focusing on survival and reproduction, we'll focus on the **erosion** of the initial state. Here's the basic algorithm:

1. **Initialization:**
   - Create a 2D grid of cells, initially filling it randomly with "rock" (1) and "air" (0). This represents the initial state of the cave.

2. **Erosion:**
   - Iterate through each cell in the grid:
      - If a "rock" cell has fewer than a certain number of "rock" neighbors, it becomes "air" (erosion). 
      - Adjust the number of required "rock" neighbors to control the density and smoothness of the cave walls.

3. **Repeat:**
   - Repeat the erosion step multiple times to gradually carve out the cave.

**1.4 Python Implementation**

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_caves(grid_size, iterations, erosion_threshold):
  """
  Generates a cave system using cellular automata.

  Args:
    grid_size: The size of the grid (grid_size x grid_size).
    iterations: The number of erosion iterations.
    erosion_threshold: The minimum number of rock neighbors for a rock cell to survive.

  Returns:
    A 2D numpy array representing the cave system.
  """

  # Initialize the grid with random values (0 for air, 1 for rock)
  grid = np.random.randint(0, 2, size=(grid_size, grid_size))

  # Perform erosion iterations
  for _ in range(iterations):
    new_grid = grid.copy()
    for i in range(1, grid_size - 1):
      for j in range(1, grid_size - 1):
        neighbors = grid[i-1:i+2, j-1:j+2].sum() - grid[i, j]  # Sum of neighbors
        if grid[i, j] == 1 and neighbors < erosion_threshold: 
          new_grid[i, j] = 0  # Erode the rock cell
    grid = new_grid

  return grid

# Generate a cave system
grid_size = 100
iterations = 50
erosion_threshold = 4
cave_map = generate_caves(grid_size, iterations, erosion_threshold)

# Visualize the cave system
plt.imshow(cave_map, cmap='gray')
plt.title("Cave System Generated with Cellular Automata")
plt.axis('off')
plt.show()
```

**1.5 Experimentation**

This code provides a basic framework. You can experiment with the following parameters to create different cave styles:

* **`iterations`:** Increase the number of iterations for more erosion and larger caves.
* **`erosion_threshold`:** Adjust the threshold to control the density of the cave walls. Lower values create more open caves, while higher values result in denser, more intricate structures.
* **Initial grid:** Try different initial grid configurations (e.g., more or less rock) to achieve varied cave shapes.

**1.6 Conclusion**

This chapter demonstrates a simple yet effective method for generating cave systems using cellular automata. By adjusting the parameters and exploring variations of the erosion rules, you can create a wide range of unique and visually appealing cave structures for your game. 

**Note:** This is a simplified example. For more complex cave systems, you can incorporate additional features such as:

* **Multiple erosion rules:** Implement different erosion rules for different areas of the cave.
* **Water level:** Simulate water levels within the caves.
* **Cave decorations:** Add features like stalactites, stalagmites, and mineral deposits.

This provides a solid foundation for further experimentation and exploration of cave generation with cellular automata in your game development projects.


