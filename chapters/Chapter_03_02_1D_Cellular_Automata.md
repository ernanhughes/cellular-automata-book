### **1D Cellular Automata Simulator**

#### **Introduction to Elementary Cellular Automata**

In elementary cellular automata:

- **Cells**: Arranged in a one-dimensional array.
- **States**: Each cell can be either **0** (white) or **1** (black).
- **Rules**: Define how cells update their states based on their current state and the state of their two immediate neighbors (left and right).
- **Rule Number**: There are \(2^{(2^3)} = 256\) possible rules, each identified by a rule number from 0 to 255.

#### **Import Necessary Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from IPython.display import display
```

- **`numpy`**: For efficient numerical computations.
- **`matplotlib`**: For plotting and visualization.
- **`ipywidgets`**: For creating interactive widgets.
- **`IPython.display`**: For displaying widgets and outputs in Jupyter.

#### **Define the Cellular Automaton Class**

Let's create a class `CellularAutomaton1D` that encapsulates the functionality to generate any 1D cellular automaton based on a specified rule number.

```python
class CellularAutomaton1D:
    def __init__(self, rule_number, size=100, steps=100, init_cond='single'):
        self.rule_number = rule_number  # Rule number (0-255)
        self.size = size                # Number of cells in a generation
        self.steps = steps              # Number of generations to simulate
        self.init_cond = init_cond      # Initial condition ('single', 'random', or custom array)
        self.rule_binary = np.array(self._decimal_to_binary(rule_number))
        self.grid = np.zeros((steps, size), dtype=np.int8)
        self.current_generation = np.zeros(size, dtype=np.int8)
        self._init_first_generation()
    
    def _decimal_to_binary(self, n):
        """Convert a decimal number to an 8-bit binary representation."""
        return np.array([int(x) for x in np.binary_repr(n, width=8)])
    
    def _init_first_generation(self):
        """Initialize the first generation based on the initial condition."""
        if self.init_cond == 'single':
            # Single cell in the middle set to 1
            self.current_generation[self.size // 2] = 1
        elif self.init_cond == 'random':
            # Random initial state
            self.current_generation = np.random.choice([0, 1], size=self.size)
        elif isinstance(self.init_cond, np.ndarray) and len(self.init_cond) == self.size:
            # Custom initial state provided as an array
            self.current_generation = self.init_cond.copy()
        else:
            raise ValueError("Invalid initial condition.")
        self.grid[0] = self.current_generation.copy()
    
    def _get_neighborhood(self, i):
        """Get the neighborhood of cell i with periodic boundary conditions."""
        left = self.current_generation[(i - 1) % self.size]
        center = self.current_generation[i]
        right = self.current_generation[(i + 1) % self.size]
        return left, center, right
    
    def _apply_rule(self, neighborhood):
        """Determine the new state of a cell based on its neighborhood and the rule."""
        # Convert the neighborhood to an index (0-7)
        idx = 7 - int(''.join(str(bit) for bit in neighborhood), 2)
        return self.rule_binary[idx]
    
    def run(self):
        """Run the cellular automaton simulation."""
        for step in range(1, self.steps):
            new_generation = np.zeros(self.size, dtype=np.int8)
            for i in range(self.size):
                neighborhood = self._get_neighborhood(i)
                new_state = self._apply_rule(neighborhood)
                new_generation[i] = new_state
            self.current_generation = new_generation
            self.grid[step] = self.current_generation.copy()
    
    def display(self):
        """Display the simulation results."""
        plt.figure(figsize=(12, 6))
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.title(f'Rule {self.rule_number}')
        plt.xlabel('Cell Index')
        plt.ylabel('Generation')
        plt.show()
```

#### **Explanation of the Code**

- **`__init__` Method**: Initializes the automaton with a rule number, grid size, number of steps, and initial condition.
    - **`rule_number`**: The ECA rule number (0-255).
    - **`size`**: Number of cells in each generation.
    - **`steps`**: Number of generations to simulate.
    - **`init_cond`**: Initial condition; can be `'single'`, `'random'`, or a custom NumPy array.
    - **`rule_binary`**: An 8-bit binary representation of the rule number.
    - **`grid`**: A 2D NumPy array to store the state of each cell at each generation.
    - **`current_generation`**: The state of the cells at the current generation.

- **`_decimal_to_binary` Method**: Converts the rule number into its 8-bit binary representation, which represents the output for each possible neighborhood.

- **`_init_first_generation` Method**: Initializes the first generation based on the specified initial condition.

- **`_get_neighborhood` Method**: Retrieves the states of the left, center, and right neighbors for a given cell index, with periodic boundary conditions.

- **`_apply_rule` Method**: Determines the new state of a cell by mapping its neighborhood to the corresponding output in the rule binary array.

- **`run` Method**: Executes the simulation by updating the state of each cell at each generation according to the rule.

- **`display` Method**: Visualizes the simulation using `matplotlib`.

#### **Using the Cellular Automaton Simulator**

Let's use this class to simulate and visualize different cellular automata.

##### **Example 1: Rule 30 with Single Cell Initial Condition**

```python
# Create an instance with Rule 30 and single cell initial condition
ca_rule_30 = CellularAutomaton1D(rule_number=30, size=100, steps=100, init_cond='single')
ca_rule_30.run()
ca_rule_30.display()
```

![Rule 30 Visualization](attachment:rule30.png)

##### **Example 2: Rule 110 with Random Initial Condition**

```python
# Create an instance with Rule 110 and random initial condition
ca_rule_110 = CellularAutomaton1D(rule_number=110, size=100, steps=100, init_cond='random')
ca_rule_110.run()
ca_rule_110.display()
```

![Rule 110 Visualization](attachment:rule110.png)

##### **Example 3: Rule 184 with Custom Initial Condition**

```python
# Custom initial condition: alternating ones and zeros
custom_init = np.array([i % 2 for i in range(100)])

# Create an instance with Rule 184 and custom initial condition
ca_rule_184 = CellularAutomaton1D(rule_number=184, size=100, steps=100, init_cond=custom_init)
ca_rule_184.run()
ca_rule_184.display()
```

![Rule 184 Visualization](attachment:rule184.png)

#### **Interactive Exploration with `ipywidgets`**

To make the exploration more dynamic, let's create an interactive widget that allows you to adjust the rule number and initial condition in real-time.

```python
def interactive_ca(rule_number, init_cond):
    ca = CellularAutomaton1D(rule_number=rule_number, size=100, steps=100, init_cond=init_cond)
    ca.run()
    ca.display()

interact(
    interactive_ca,
    rule_number=IntSlider(min=0, max=255, step=1, value=30, description='Rule Number'),
    init_cond=['single', 'random']
)
```

- **`interact` Function**: Creates interactive widgets for the parameters `rule_number` and `init_cond`.
- **Parameters**:
    - **`rule_number`**: Slider to select any rule number between 0 and 255.
    - **`init_cond`**: Dropdown to select between `'single'` and `'random'` initial conditions.

#### **Understanding the Rule Encoding**

Each rule number corresponds to a unique set of outputs for all possible neighborhoods. The neighborhoods are the combinations of the states of a cell and its immediate neighbors:

| Neighborhood | Binary Index | New State |
|--------------|--------------|-----------|
| 111          | 7            | Rule[7]   |
| 110          | 6            | Rule[6]   |
| 101          | 5            | Rule[5]   |
| 100          | 4            | Rule[4]   |
| 011          | 3            | Rule[3]   |
| 010          | 2            | Rule[2]   |
| 001          | 1            | Rule[1]   |
| 000          | 0            | Rule[0]   |

For example, Rule 30 has a binary representation of `00011110`, which maps to the new states for each neighborhood accordingly.

#### **Customizing the Simulator**

You can extend the simulator to include more features:

- **Adjustable Grid Size and Steps**: Change `size` and `steps` parameters to simulate larger grids or more generations.
- **Non-Periodic Boundary Conditions**: Modify `_get_neighborhood` method to implement fixed boundary conditions.
- **Multiple States**: Extend the automaton to support more than two states per cell.

#### **Saving the Simulation Results**

You can save the grid for further analysis or export the visualization.

```python
# Save the grid data
np.save('ca_rule_30.npy', ca_rule_30.grid)

# Save the visualization
plt.imsave('ca_rule_30.png', ca_rule_30.grid, cmap='binary')
```

#### **Analyzing the Automaton**

Compute statistical properties or patterns:

```python
# Calculate the density of active cells over time
density = np.mean(ca_rule_30.grid, axis=1)
plt.figure(figsize=(10, 4))
plt.plot(density)
plt.title('Density of Active Cells Over Time')
plt.xlabel('Generation')
plt.ylabel('Density')
plt.show()
```


### Analyzing Patterns and Properties

1. **Stable Patterns**: Rules like 0 and 255 lead to stable configurations (e.g., all cells are off or on).
2. **Chaotic Behavior**: Rules like 30 and 45 produce chaotic, unpredictable patterns.
3. **Complexity and Universality**: Rule 110 is Turing complete, meaning it can simulate any computation given enough time and space.
