The Adaptive Step Search (ASS) algorithm is a heuristic optimization technique that adapts its search steps based on feedback from the objective function. Here's a Python implementation:

### Adaptive Step Searching Algorithm in Python
```python
import numpy as np

def adaptive_step_search(objective_function, bounds, max_iter=1000, initial_step_size=0.1, step_decay=0.9, tolerance=1e-6):
    """
    Adaptive Step Searching Algorithm
    
    Args:
        objective_function (function): The function to minimize.
        bounds (list of tuples): Search bounds for each dimension [(min1, max1), (min2, max2), ...].
        max_iter (int): Maximum number of iterations.
        initial_step_size (float): Initial step size for the search.
        step_decay (float): Factor by which to reduce the step size.
        tolerance (float): Convergence tolerance for the function value.
    
    Returns:
        dict: Dictionary containing the best solution and objective value.
    """
    # Initialize variables
    dim = len(bounds)
    current_solution = np.array([np.random.uniform(low, high) for low, high in bounds])
    current_value = objective_function(current_solution)
    step_size = initial_step_size

    best_solution = current_solution
    best_value = current_value

    for iteration in range(max_iter):
        # Generate candidate solutions around the current solution
        candidates = [
            current_solution + step_size * np.random.uniform(-1, 1, dim),
            current_solution - step_size * np.random.uniform(-1, 1, dim),
        ]

        # Evaluate candidate solutions
        for candidate in candidates:
            # Ensure the candidate is within bounds
            candidate = np.clip(candidate, [low for low, _ in bounds], [high for _, high in bounds])
            candidate_value = objective_function(candidate)

            # Update best solution if candidate is better
            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value
                current_solution = candidate
                current_value = candidate_value
                break
        else:
            # If no improvement, reduce step size
            step_size *= step_decay

        # Convergence check
        if step_size < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return {
        "best_solution": best_solution,
        "best_value": best_value
    }

# Example Usage
def example_function(x):
    """Example objective function: Sphere function."""
    return np.sum(x**2)

# Define bounds for the problem
bounds = [(-5, 5), (-5, 5)]  # Example 2D problem

result = adaptive_step_search(
    objective_function=example_function,
    bounds=bounds,
    max_iter=1000,
    initial_step_size=0.5,
    step_decay=0.9,
    tolerance=1e-6
)

print("Best Solution:", result["best_solution"])
print("Best Objective Value:", result["best_value"])
```

### Explanation:
1. **Initialization**:
   - The search starts at a random point within the defined bounds.
   - The `step_size` determines the initial exploration range.

2. **Candidate Generation**:
   - Generates candidate solutions by perturbing the current solution.

3. **Evaluation**:
   - Compares the candidate solutions to the current best.
   - Updates the best solution if a candidate performs better.

4. **Step Size Adaptation**:
   - Reduces the `step_size` if no better solution is found, allowing finer exploration.

5. **Convergence**:
   - Stops when the `step_size` falls below a specified tolerance.

### Example Output:
For the provided Sphere function \( f(x) = \sum x_i^2 \), the algorithm finds the global minimum at \( x = [0, 0] \) with \( f(x) = 0 \).

Feel free to adjust the parameters and test this implementation on your objective functions!