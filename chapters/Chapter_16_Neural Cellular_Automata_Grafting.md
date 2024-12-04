# **Neural Cellular Automata Grafting**

**Abstract**: In this chapter, we delve into the concept of Neural Cellular Automata (NCA) grafting, an advanced technique that involves combining multiple NCAs to create complex patterns and behaviors. We'll explore the theoretical foundations of NCA grafting, its applications, and provide a step-by-step guide to implementing grafting using Python and PyTorch. By the end of this chapter, you'll be equipped with the knowledge to experiment with NCA grafting in your projects.

---

## **Table of Contents**

1. [Introduction to Neural Cellular Automata Grafting](#introduction-to-neural-cellular-automata-grafting)
2. [Understanding Neural Cellular Automata](#understanding-neural-cellular-automata)
3. [What is NCA Grafting?](#what-is-nca-grafting)
4. [Applications of NCA Grafting](#applications-of-nca-grafting)
5. [Implementing NCA Grafting in Python](#implementing-nca-grafting-in-python)
    - [Setting Up the Environment](#setting-up-the-environment)
    - [Defining the NCA Model](#defining-the-nca-model)
    - [Training Multiple NCAs](#training-multiple-ncas)
    - [Grafting the NCAs](#grafting-the-ncas)
6. [Visualizing the Results](#visualizing-the-results)
7. [Experimenting with Grafting Techniques](#experimenting-with-grafting-techniques)
8. [Conclusion](#conclusion)
9. [Exercises](#exercises)
10. [Further Reading](#further-reading)

---

## **Introduction to Neural Cellular Automata Grafting**

Neural Cellular Automata Grafting is a technique that involves merging or combining different Neural Cellular Automata to create new behaviors and patterns. By grafting, we can introduce complexity and variability into the system, allowing for more dynamic and intricate simulations.

This chapter will guide you through:

- Understanding the principles of NCA grafting.
- Implementing NCA grafting in Python using PyTorch.
- Visualizing and analyzing the results.
- Experimenting with different grafting methods.

---

## **Understanding Neural Cellular Automata**

Before diving into grafting, it's essential to have a solid understanding of Neural Cellular Automata.

### **Neural Cellular Automata (NCA)**

- **Cellular Automata (CA)**: A grid of cells that evolve over discrete time steps according to local interaction rules.
- **Neural Cellular Automata**: Instead of predefined rules, NCAs use neural networks to determine the state updates of each cell based on its neighborhood.

### **Key Components of NCA**

- **Grid**: A 2D (or higher-dimensional) array representing the state of the system.
- **States**: Each cell has a state vector, often including multiple channels (e.g., RGBA, hidden states).
- **Neighborhood**: Typically includes the cell itself and its immediate neighbors.
- **Update Rule**: A neural network that computes the new state based on the neighborhood.

---

## **What is NCA Grafting?**

NCA Grafting involves combining two or more trained NCAs into a single system. This can be achieved by:

- **Spatial Grafting**: Placing different NCAs in separate regions of the grid and allowing them to interact at the boundaries.
- **Parameter Grafting**: Merging the parameters (weights) of different NCA models.
- **Dynamic Grafting**: Switching or blending update rules during the simulation.

### **Benefits of NCA Grafting**

- **Complex Patterns**: Generate more intricate and diverse patterns.
- **Adaptive Behaviors**: Simulate systems that adapt or change over time.
- **Creative Exploration**: Discover new emergent behaviors by combining different NCAs.

---

## **Applications of NCA Grafting**

- **Artistic Texture Generation**: Create unique textures by blending different patterns.
- **Biological Simulations**: Model interactions between different cell types or organisms.
- **Procedural Content Generation**: Generate complex environments or structures in games and simulations.
- **Adaptive Systems**: Develop systems that change behavior in response to environmental conditions.

---

## **Implementing NCA Grafting in Python**

In this section, we'll implement NCA grafting using PyTorch. We'll train two separate NCAs and then graft them together.

### **Setting Up the Environment**

#### **Prerequisites**

- **Python 3.7+**
- **PyTorch**
- **NumPy**
- **Matplotlib**

#### **Installing Libraries**

```bash
pip install torch torchvision matplotlib numpy
```

---

### **Defining the NCA Model**

We'll define an NCA model similar to the one used in previous chapters but with the flexibility to accommodate grafting.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5):
        super(NeuralCA, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        # Perception layers
        self.perception = nn.Conv2d(channel_n, channel_n * 3, kernel_size=3, padding=1, groups=channel_n, bias=False)
        
        # Update layers
        self.update = nn.Sequential(
            nn.Conv2d(channel_n * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, channel_n, kernel_size=1),
        )
        
        # Initialize the perception kernels
        with torch.no_grad():
            sobel_x = torch.tensor([[1.0, 0.0, -1.0],
                                    [2.0, 0.0, -2.0],
                                    [1.0, 0.0, -1.0]])
            sobel_y = torch.tensor([[1.0, 2.0, 1.0],
                                    [0.0, 0.0, 0.0],
                                    [-1.0, -2.0, -1.0]])
            laplace = torch.tensor([[0.0, 1.0, 0.0],
                                    [1.0, -4.0, 1.0],
                                    [0.0, 1.0, 0.0]])
            kernels = torch.stack([sobel_x, sobel_y, laplace])
            kernels = kernels.unsqueeze(1).unsqueeze(1) / 8.0
            self.perception.weight.data = kernels.repeat(channel_n, 1, 1, 1)
        
    def forward(self, x):
        # Perception
        y = self.perception(x)
        
        # Update
        dx = self.update(y)
        
        # Stochastic update
        mask = (torch.rand_like(x[:, :1, :, :]) <= self.fire_rate).float()
        dx = dx * mask
        
        # Apply the update
        x = x + dx
        return x
```

- **Explanation**:
  - **Perception Layer**: Uses fixed Sobel and Laplacian filters to extract features.
  - **Update Network**: A small convolutional neural network that computes state updates.
  - **Stochastic Update**: Applies updates randomly to simulate asynchronous cell updates.

---

### **Training Multiple NCAs**

We'll train two NCAs to generate different patterns or images.

#### **Step 1: Define the Training Loop**

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_nca(model, target_img, epochs=5000, lr=2e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = torch.from_numpy(target_img).permute(2, 0, 1).unsqueeze(0).to(device)
    losses = []

    # Initialize the grid
    x = torch.zeros(1, model.channel_n, target.shape[2], target.shape[3]).to(device)
    x[:, :4, target.shape[2]//2, target.shape[3]//2] = 1.0  # Seed in the center

    for epoch in tqdm(range(epochs)):
        x.requires_grad_(True)
        for _ in range(8):
            x = model(x)
            x = torch.clamp(x, 0.0, 1.0)
        loss = F.mse_loss(x[:, :4, :, :], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, x.detach(), losses
```

#### **Step 2: Prepare Target Images**

We'll use simple patterns for demonstration.

```python
# Create target images
def create_target_circle(size, radius, color):
    y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
    mask = x**2 + y**2 <= radius**2
    img = np.zeros((size, size, 4), dtype=np.float32)
    img[mask] = color
    return img

size = 64
target_img1 = create_target_circle(size, 20, [1.0, 0.0, 0.0, 1.0])  # Red circle
target_img2 = create_target_circle(size, 15, [0.0, 0.0, 1.0, 1.0])  # Blue circle
```

#### **Step 3: Train the NCAs**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train NCA1
model1 = NeuralCA().to(device)
model1, final_state1, losses1 = train_nca(model1, target_img1, device=device)

# Train NCA2
model2 = NeuralCA().to(device)
model2, final_state2, losses2 = train_nca(model2, target_img2, device=device)
```

---

### **Grafting the NCAs**

Now that we have two trained NCAs, we'll graft them together.

#### **Method 1: Spatial Grafting**

We'll initialize a grid where one half is updated by `model1` and the other half by `model2`.

```python
def graft_models(model1, model2, size, steps, device='cpu'):
    # Initialize the grid
    x = torch.zeros(1, model1.channel_n, size, size).to(device)
    x[:, :4, :, :] = 0.5  # Initial state

    # Create a mask to determine which model updates which region
    mask1 = torch.zeros(1, 1, size, size).to(device)
    mask1[:, :, :, :size//2] = 1.0  # Left half for model1
    mask2 = 1.0 - mask1  # Right half for model2

    x_history = []

    for _ in range(steps):
        x1 = model1(x)
        x2 = model2(x)
        x = x1 * mask1 + x2 * mask2 + x * (1 - mask1 - mask2)
        x = torch.clamp(x, 0.0, 1.0)
        x_history.append(x.detach().cpu().numpy())
    
    return x_history
```

- **Explanation**:
  - **Masking**: We create binary masks to specify regions updated by each model.
  - **Combining Updates**: We apply updates from both models to their respective regions.

#### **Run the Grafting Simulation**

```python
x_history = graft_models(model1, model2, size=64, steps=200, device=device)
```

---

## **Visualizing the Results**

We'll create an animation to visualize how the grafted NCA evolves.

```python
from matplotlib import animation
from IPython.display import HTML

def visualize_grafting(x_history):
    fig, ax = plt.subplots()
    ims = []

    for x in x_history[::5]:
        img = x[0, :4].transpose(1, 2, 0)
        im = ax.imshow(img)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())

visualize_grafting(x_history)
```

- **Explanation**:
  - **Visualization**: Displays the evolution of the grid over time.
  - **Observation**: You can observe how the two NCAs interact at the boundary.

---

## **Experimenting with Grafting Techniques**

### **Method 2: Parameter Grafting**

Combine the parameters of the two models.

#### **Combine Model Parameters**

```python
def average_parameters(model1, model2):
    model_combined = NeuralCA().to(device)
    with torch.no_grad():
        for p1, p2, pc in zip(model1.parameters(), model2.parameters(), model_combined.parameters()):
            pc.copy_((p1 + p2) / 2.0)
    return model_combined

model_combined = average_parameters(model1, model2)
```

#### **Run the Combined Model**

```python
# Initialize the grid
x = torch.zeros(1, model_combined.channel_n, size, size).to(device)
x[:, :4, size//2, size//2] = 1.0  # Seed in the center

# Evolve the grid
x_history_combined = []
for _ in range(200):
    x = model_combined(x)
    x = torch.clamp(x, 0.0, 1.0)
    x_history_combined.append(x.detach().cpu().numpy())
```

#### **Visualize the Combined Model**

```python
visualize_grafting(x_history_combined)
```

### **Method 3: Dynamic Grafting**

Switch between models during simulation.

```python
def dynamic_grafting(model1, model2, size, steps, switch_step, device='cpu'):
    # Initialize the grid
    x = torch.zeros(1, model1.channel_n, size, size).to(device)
    x[:, :4, size//2, size//2] = 1.0  # Seed in the center

    x_history = []

    for step in range(steps):
        if step < switch_step:
            x = model1(x)
        else:
            x = model2(x)
        x = torch.clamp(x, 0.0, 1.0)
        x_history.append(x.detach().cpu().numpy())
    
    return x_history

x_history_dynamic = dynamic_grafting(model1, model2, size=64, steps=200, switch_step=100, device=device)
```

#### **Visualize Dynamic Grafting**

```python
visualize_grafting(x_history_dynamic)
```

- **Explanation**:
  - **Switch Step**: Determines when to switch from one model to the other.
  - **Observation**: The grid may exhibit different behaviors before and after the switch.

---

## **Conclusion**

Neural Cellular Automata Grafting offers a powerful technique to explore complex behaviors by combining different NCAs. Through spatial, parameter, and dynamic grafting methods, we can create rich, emergent patterns that are not achievable with a single NCA alone.

**Key Takeaways**:

- **Spatial Grafting**: Combining NCAs in different regions allows for interaction at boundaries.
- **Parameter Grafting**: Merging model parameters can create hybrid behaviors.
- **Dynamic Grafting**: Switching update rules during simulation introduces temporal complexity.

By experimenting with these grafting techniques, you can unlock new possibilities in procedural generation, simulations, and creative explorations.

---

## **Exercises**

1. **Experiment with Different Patterns**: Use different target images for training the NCAs (e.g., shapes, letters) and observe how grafting affects the results.

2. **Adjust Fire Rates**: Modify the `fire_rate` parameter for each model and see how it influences the grafted system.

3. **Non-Binary Masks**: Instead of binary masks, use gradient masks to blend updates smoothly between models.

4. **Train a Combined Model**: After grafting, continue training the combined model and observe if it learns new patterns.

5. **3D NCA Grafting**: Extend the grafting techniques to 3D NCAs and visualize the results.

---

## **Further Reading**

- **"Growing Neural Cellular Automata" by Mordvintsev et al.**: The foundational paper introducing NCAs for image generation.

- **"Self-Organizing Textures"**: Research on using NCAs for generating self-organizing patterns and textures.

- **GitHub Repositories**:
  - [NCA Implementations](https://github.com/google-research/self-organising-systems)
  - [Community Projects on NCAs](https://github.com/topics/neural-cellular-automata)

- **Articles and Tutorials**:
  - [Distill Article on NCAs](https://distill.pub/2020/growing-ca/): An interactive exploration of NCAs.
  - [Blog Posts on NCA Grafting](https://blog.otoro.net/): Discussions on advanced NCA techniques.

---

By incorporating Neural Cellular Automata Grafting into your programming toolkit, you open up avenues for creating systems with unprecedented complexity and adaptability. Whether for artistic endeavors, scientific simulations, or exploring the boundaries of artificial life, NCA grafting provides a rich field for experimentation and discovery.

---

**Note**: Ensure you have the necessary libraries installed and that you run the code in an environment that supports GPU acceleration for optimal performance (e.g., a Jupyter Notebook with CUDA-enabled PyTorch).