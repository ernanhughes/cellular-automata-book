# **Procedural Generation Using Constraint Satisfaction**

Procedural generation is a method of creating data algorithmically rather than manually, often used in computer graphics, games, and simulations to generate content like terrains, levels, and puzzles. Constraint Satisfaction Problems (CSPs) provide a powerful framework for procedural generation by defining variables, domains, and constraints that the generated content must satisfy.

In this chapter, we'll explore how to use constraint satisfaction in Python for procedural generation. We'll walk through a practical example using the `python-constraint` library to generate a simple puzzle procedurally. By the end of this chapter, you'll have a solid understanding of CSPs and how to apply them to generate content dynamically.

---

## **Table of Contents**

1. [Introduction to Constraint Satisfaction Problems](#introduction-to-constraint-satisfaction-problems)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Example: Procedural Generation of a Magic Square](#example-procedural-generation-of-a-magic-square)
    - [Understanding Magic Squares](#understanding-magic-squares)
    - [Defining the CSP](#defining-the-csp)
    - [Implementing the CSP in Python](#implementing-the-csp-in-python)
4. [Visualizing the Generated Magic Square](#visualizing-the-generated-magic-square)
5. [Extending the Example](#extending-the-example)
6. [Conclusion](#conclusion)
7. [Exercises](#exercises)
8. [Further Reading](#further-reading)

---

## **Introduction to Constraint Satisfaction Problems**

A **Constraint Satisfaction Problem** consists of:

- **Variables**: Elements that need to be assigned values.
- **Domains**: Possible values that each variable can take.
- **Constraints**: Rules that restrict the values the variables can simultaneously take.

CSPs aim to find an assignment of values to variables that satisfies all constraints. They are widely used in artificial intelligence for tasks like scheduling, planning, and puzzle solving.

---

## **Setting Up the Environment**

Before we begin, ensure that you have the necessary Python libraries installed. We'll use the `python-constraint` library for defining and solving the CSP.

### **Install Required Libraries**

```bash
pip install python-constraint
```

---

## **Example: Procedural Generation of a Magic Square**

### **Understanding Magic Squares**

A **Magic Square** is an \( n \times n \) grid filled with distinct positive integers in the range \( 1 \) to \( n^2 \) such that each cell contains a different integer and the sum of the integers in each row, column, and the two main diagonals is the same.

For a \( 3 \times 3 \) magic square, the magic constant (the common sum) is:

\[ M = \frac{n(n^2 + 1)}{2} = 15 \]

### **Defining the CSP**

- **Variables**: Each cell in the \( 3 \times 3 \) grid.
- **Domains**: Possible values from \( 1 \) to \( 9 \).
- **Constraints**:
    - All values are distinct.
    - The sum of each row, column, and diagonal equals \( 15 \).

### **Implementing the CSP in Python**

Let's implement this step by step.

#### **Step 1: Import Libraries**

```python
from constraint import Problem, AllDifferentConstraint, ExactSumConstraint
import numpy as np
```

#### **Step 2: Initialize the Problem**

```python
problem = Problem()
```

#### **Step 3: Define Variables and Domains**

We will represent the cells as variables named from \( 'A1' \) to \( 'C3' \), where the letter represents the row and the number represents the column.

```python
cells = ['A1', 'A2', 'A3',
         'B1', 'B2', 'B3',
         'C1', 'C2', 'C3']

# The domain is numbers from 1 to 9
domain = range(1, 10)

problem.addVariables(cells, domain)
```

#### **Step 4: Add Constraints**

##### **AllDifferent Constraint**

All numbers must be distinct.

```python
problem.addConstraint(AllDifferentConstraint(), cells)
```

##### **Sum Constraints for Rows**

```python
# Rows
problem.addConstraint(ExactSumConstraint(15), ['A1', 'A2', 'A3'])
problem.addConstraint(ExactSumConstraint(15), ['B1', 'B2', 'B3'])
problem.addConstraint(ExactSumConstraint(15), ['C1', 'C2', 'C3'])
```

##### **Sum Constraints for Columns**

```python
# Columns
problem.addConstraint(ExactSumConstraint(15), ['A1', 'B1', 'C1'])
problem.addConstraint(ExactSumConstraint(15), ['A2', 'B2', 'C2'])
problem.addConstraint(ExactSumConstraint(15), ['A3', 'B3', 'C3'])
```

##### **Sum Constraints for Diagonals**

```python
# Diagonals
problem.addConstraint(ExactSumConstraint(15), ['A1', 'B2', 'C3'])
problem.addConstraint(ExactSumConstraint(15), ['A3', 'B2', 'C1'])
```

#### **Step 5: Solve the Problem**

We can now solve the problem and retrieve the solutions.

```python
solutions = problem.getSolutions()
```

#### **Step 6: Display a Solution**

For the sake of this example, we'll display one of the solutions.

```python
if solutions:
    solution = solutions[0]  # Get the first solution
    print("One possible Magic Square:")
    print(f"{solution['A1']} {solution['A2']} {solution['A3']}")
    print(f"{solution['B1']} {solution['B2']} {solution['B3']}")
    print(f"{solution['C1']} {solution['C2']} {solution['C3']}")
else:
    print("No solution found.")
```

---

### **Full Code in a Jupyter Notebook Cell**

```python
from constraint import Problem, AllDifferentConstraint, ExactSumConstraint

# Initialize the problem
problem = Problem()

# Define variables
cells = ['A1', 'A2', 'A3',
         'B1', 'B2', 'B3',
         'C1', 'C2', 'C3']

domain = range(1, 10)
problem.addVariables(cells, domain)

# Add constraints
problem.addConstraint(AllDifferentConstraint(), cells)

# Rows
problem.addConstraint(ExactSumConstraint(15), ['A1', 'A2', 'A3'])
problem.addConstraint(ExactSumConstraint(15), ['B1', 'B2', 'B3'])
problem.addConstraint(ExactSumConstraint(15), ['C1', 'C2', 'C3'])

# Columns
problem.addConstraint(ExactSumConstraint(15), ['A1', 'B1', 'C1'])
problem.addConstraint(ExactSumConstraint(15), ['A2', 'B2', 'C2'])
problem.addConstraint(ExactSumConstraint(15), ['A3', 'B3', 'C3'])

# Diagonals
problem.addConstraint(ExactSumConstraint(15), ['A1', 'B2', 'C3'])
problem.addConstraint(ExactSumConstraint(15), ['A3', 'B2', 'C1'])

# Solve the problem
solutions = problem.getSolutions()

# Display a solution
if solutions:
    solution = solutions[0]
    print("One possible Magic Square:")
    print(f"{solution['A1']} {solution['A2']} {solution['A3']}")
    print(f"{solution['B1']} {solution['B2']} {solution['B3']}")
    print(f"{solution['C1']} {solution['C2']} {solution['C3']}")
else:
    print("No solution found.")
```

**Output:**

```
One possible Magic Square:
2 7 6
9 5 1
4 3 8
```

---

## **Visualizing the Generated Magic Square**

To enhance the presentation, let's visualize the magic square using Matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt

def display_magic_square(solution):
    # Convert the solution dictionary to a 2D list
    magic_square = [
        [solution['A1'], solution['A2'], solution['A3']],
        [solution['B1'], solution['B2'], solution['B3']],
        [solution['C1'], solution['C2'], solution['C3']]
    ]
    magic_square = np.array(magic_square)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(np.zeros((3, 3)), cmap='Pastel1')
    
    for (i, j), value in np.ndenumerate(magic_square):
        ax.text(j, i, f'{value}', va='center', ha='center', fontsize=24)
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# Display the magic square
display_magic_square(solution)
```

This code creates a simple visualization of the magic square, displaying the numbers in a grid.

---

## **Extending the Example**

### **Generating Larger Magic Squares**

You can extend the code to generate larger magic squares (e.g., \( 4 \times 4 \)). However, keep in mind that the number of variables and constraints increases significantly, which can impact performance.

#### **Example: \( 4 \times 4 \) Magic Square**

The magic constant for a \( 4 \times 4 \) magic square is:

\[ M = \frac{n(n^2 + 1)}{2} = 34 \]

The implementation would involve:

- Defining 16 variables (cells from 'A1' to 'D4').
- Setting the domain to numbers from 1 to 16.
- Adding constraints for all rows, columns, and diagonals to sum to 34.
- Ensuring all numbers are distinct.

### **Procedural Generation of Sudoku Puzzles**

Sudoku puzzles are another excellent example of using CSPs for procedural generation.

- **Variables**: Each cell in the \( 9 \times 9 \) grid.
- **Domains**: Numbers from 1 to 9.
- **Constraints**:
    - Each row contains all numbers from 1 to 9 without repetition.
    - Each column contains all numbers from 1 to 9 without repetition.
    - Each \( 3 \times 3 \) subgrid contains all numbers from 1 to 9 without repetition.

Implementing Sudoku generation involves more complex constraints but follows similar principles.

---

## **Conclusion**

In this chapter, we've explored how constraint satisfaction can be leveraged for procedural generation. By defining variables, domains, and constraints, we can generate content that meets specific criteria without manually crafting each instance.

Key takeaways:

- **CSPs** provide a structured way to define and solve problems with constraints.
- **Procedural Generation** benefits from CSPs by automating the creation of content that adheres to specified rules.
- **Python** offers libraries like `python-constraint` that simplify the implementation of CSPs.

---

## **Exercises**

1. **Extend to \( 4 \times 4 \) Magic Square**: Modify the code to generate a \( 4 \times 4 \) magic square. Note the increased complexity and how it affects performance.

2. **Sudoku Puzzle Generation**: Implement a CSP to generate a complete Sudoku puzzle. Then, remove some numbers to create a playable puzzle.

3. **Crossword Puzzle Generation**: Use CSPs to generate a simple crossword puzzle by fitting words into a grid based on given constraints.

4. **Map Generation**: Procedurally generate a map where regions are colored such that no adjacent regions share the same color (the map coloring problem).

5. **Timetable Scheduling**: Use CSPs to create a timetable that schedules classes without conflicts, considering constraints like room availability and instructor schedules.

---

## **Further Reading**

- **"Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig**: A comprehensive resource on AI, including CSPs.

- **Constraint Programming**: Explore more about constraint programming paradigms and their applications.

- **Python Constraint Documentation**: Visit the [official documentation](https://labix.org/python-constraint) for more details and advanced usage.

- **Procedural Content Generation in Games**: Research how CSPs are used in game development for content generation.

---

By integrating CSPs into your procedural generation toolkit, you unlock the ability to create complex, rule-abiding content algorithmically. This approach not only saves time but also opens up possibilities for creating dynamic and adaptive systems in your programs.

---

**Note**: The code provided is intended to be run in a Jupyter Notebook cell. Ensure all necessary libraries are installed and imported as shown.