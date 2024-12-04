# **Texture Generation with Neural Cellular Automata**

**Abstract**: In this chapter, we explore the fascinating world of Neural Cellular Automata (NCA) for texture generation. NCAs combine principles from cellular automata and neural networks to create complex, self-organizing patterns. We'll delve into how NCAs work, their applications in texture generation, and provide a step-by-step guide to implementing them using Python in a Jupyter Notebook.

---

## **Table of Contents**

1. [Introduction to Neural Cellular Automata](#introduction-to-neural-cellular-automata)
2. [Understanding Cellular Automata](#understanding-cellular-automata)
3. [Neural Networks in Cellular Automata](#neural-networks-in-cellular-automata)
4. [Texture Generation with NCA](#texture-generation-with-nca)
5. [Implementing NCA in Python](#implementing-nca-in-python)
    - [Setting Up the Environment](#setting-up-the-environment)
    - [Defining the Neural Cellular Automaton](#defining-the-neural-cellular-automaton)
    - [Training the NCA](#training-the-nca)
    - [Generating Textures](#generating-textures)
6. [Visualizing the Results](#visualizing-the-results)
7. [Extending the Model](#extending-the-model)
8. [Conclusion](#conclusion)
9. [Exercises](#exercises)
10. [Further Reading](#further-reading)

---

## **Introduction to Neural Cellular Automata**

Neural Cellular Automata (NCA) are a blend of traditional cellular automata and neural networks. They model complex systems where local interactions between cells lead to emergent global patterns. NCAs have gained attention for their ability to generate intricate textures and images, simulate natural phenomena, and even perform computations.

### **Key Concepts**

- **Cellular Automata (CA)**: A grid of cells that evolve over time based on predefined rules and the states of neighboring cells.
- **Neural Networks**: Computational models inspired by biological neural networks, capable of learning patterns from data.
- **Neural Cellular Automata**: CAs where the update rules are parameterized by neural networks, allowing for learned, flexible behaviors.

---

## **Understanding Cellular Automata**

Before diving into NCAs, it's essential to understand the fundamentals of cellular automata.

### **Basic Structure**

- **Grid**: Typically a 2D array of cells.
- **States**: Each cell has a state, often represented by integers or colors.
- **Neighborhood**: The set of neighboring cells influencing a cell's next state (e.g., Moore or Von Neumann neighborhoods).
- **Rules**: Deterministic or probabilistic functions dictating state transitions based on neighbor states.

### **Examples**

- **Conway's Game of Life**: A classic example where simple rules lead to complex patterns.
- **Elementary Cellular Automata**: 1D CAs with two possible states per cell and simple rule sets.

---

## **Neural Networks in Cellular Automata**

Introducing neural networks into cellular automata allows for:

- **Learnable Rules**: Instead of hardcoded rules, the CA updates are learned from data.
- **Complex Behaviors**: Ability to capture intricate patterns and adapt to different tasks.
- **Generalization**: Neural networks can generalize beyond the training data, leading to novel pattern generation.

---

## **Texture Generation with NCA**

Textures are essential in computer graphics for adding realism to 3D models and environments. NCAs can generate textures by learning to produce desired patterns through local interactions.

### **Applications**

- **Procedural Texture Generation**: Creating textures algorithmically without manual design.
- **Style Transfer**: Applying the style of one image to another.
- **Simulation of Natural Patterns**: Generating textures resembling wood grain, marble, or organic tissues.

---

## **Implementing NCA in Python**

In this section, we'll implement a Neural Cellular Automaton to generate textures. We'll use PyTorch, a popular deep learning library, for the neural network components.

### **Setting Up the Environment**

#### **Prerequisites**

- **Python 3.7+**
- **PyTorch**
- **Matplotlib**
- **NumPy**

#### **Installing Libraries**

```bash
pip install torch torchvision matplotlib numpy
```

---

### **Defining the Neural Cellular Automaton**

We'll create a neural network that updates the state of each cell based on its neighbors.

#### **Step 1: Import Libraries**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
```

#### **Step 2: Define the NCA Model**

```python
class NCAModel(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5):
        super(NCAModel, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.conv1 = nn.Conv2d(channel_n * 3, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, channel_n, kernel_size=1)
    
    def perceive(self, x, device):
        """Perceive the neighboring cells."""
        kernel = torch.tensor([
            [[0.0, 1.0, 0.0],
             [1.0, 0.0, 1.0],
             [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 1.0]],
            [[0.0, -1.0, 0.0],
             [-1.0, 0.0, -1.0],
             [0.0, -1.0, 0.0]]
        ]).to(device)
        kernel = kernel.unsqueeze(1).expand(-1, self.channel_n, -1, -1)
        y = F.conv2d(x, kernel, padding=1, groups=self.channel_n)
        return y

    def forward(self, x, device):
        y = self.perceive(x, device)
        x = torch.cat([x, y], 1)
        x = F.relu(self.conv1(x))
        dx = self.conv2(x)
        stochastic_mask = (torch.rand(dx.shape[0], 1, dx.shape[2], dx.shape[3]) > self.fire_rate).to(device)
        dx = dx * stochastic_mask
        return dx
```

- **Explanation**:
  - **Perception**: The `perceive` method computes the gradients in different directions.
  - **Forward Pass**: Combines the current state and perceived information, processes through convolutional layers, and applies a stochastic update mask.

---

### **Training the NCA**

We'll train the NCA to generate a target texture.

#### **Step 1: Load the Target Texture**

```python
from PIL import Image

def load_target_image(path, size):
    img = Image.open(path)
    img = img.resize((size, size))
    img = np.array(img) / 255.0
    return img

target_img = load_target_image('texture.png', 64)
plt.imshow(target_img)
plt.axis('off')
plt.show()
```

- **Note**: Ensure you have a `texture.png` image in your working directory.

#### **Step 2: Prepare Training Data**

```python
target = torch.from_numpy(target_img).permute(2, 0, 1).unsqueeze(0).float()
```

#### **Step 3: Define the Training Loop**

```python
def train_nca(model, target, device, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Initialize the grid
    grid_size = target.shape[-1]
    x = torch.zeros(1, model.channel_n, grid_size, grid_size).to(device)
    x[:, :4, grid_size//2, grid_size//2] = 1.0  # Seed in the center
    
    losses = []
    
    for epoch in range(epochs):
        x.requires_grad_(True)
        dx = model(x, device)
        x = x + dx
        x = torch.clamp(x, 0.0, 1.0)
        
        loss = loss_fn(x[:, :4, :, :], target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return x, losses
```

- **Explanation**:
  - **Optimizer**: Adam optimizer for training.
  - **Loss Function**: Mean Squared Error between the generated and target textures.
  - **Grid Initialization**: Start with a grid seeded at the center.
  - **Training Loop**: For each epoch, perform forward pass, compute loss, backpropagate, and update the model.

#### **Step 4: Run the Training**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NCAModel(channel_n=16, fire_rate=0.5).to(device)
trained_x, losses = train_nca(model, target, device, epochs=1000, lr=1e-3)
```

---

### **Generating Textures**

After training, we can use the model to generate textures starting from random seeds.

#### **Step 1: Initialize a Random Grid**

```python
def initialize_grid(model, size, device):
    x = torch.zeros(1, model.channel_n, size, size).to(device)
    x[:, :4, size//2, size//2] = 1.0  # Seed in the center
    return x

grid_size = 64
x = initialize_grid(model, grid_size, device)
```

#### **Step 2: Evolve the Grid**

```python
def evolve_nca(model, x, steps, device):
    x_history = []
    for _ in range(steps):
        dx = model(x, device)
        x = x + dx
        x = torch.clamp(x, 0.0, 1.0)
        x_history.append(x.detach().cpu().numpy())
    return x_history

x_history = evolve_nca(model, x, steps=200, device=device)
```

---

## **Visualizing the Results**

We'll create an animation to visualize how the texture evolves over time.

```python
from matplotlib import animation

fig, ax = plt.subplots()
ims = []

for i in range(0, len(x_history), 5):
    im = ax.imshow(x_history[i][0, :4].transpose(1, 2, 0))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
plt.show()
```

- **Explanation**:
  - **Animation**: We collect frames at intervals to create an animation.
  - **Visualization**: Displays the evolution of the texture over time.

---

## **Extending the Model**

### **Different Textures**

You can experiment with different target textures to see how the NCA learns to generate various patterns.

### **Adjusting Parameters**

- **Fire Rate**: Controls the sparsity of updates. Lower values lead to sparser updates.
- **Channel Count**: Increasing the number of channels can capture more complex patterns.
- **Learning Rate**: Adjusting the learning rate can impact training stability.

### **Additional Features**

- **Stochastic Updates**: Introduce randomness to simulate natural variability.
- **Boundary Conditions**: Implement different boundary conditions (e.g., wrap-around) for varied behaviors.

---

## **Conclusion**

Neural Cellular Automata offer a powerful approach to procedural texture generation. By leveraging neural networks within the cellular automata framework, we can create complex, self-organizing patterns that are both visually appealing and computationally efficient.

---

## **Exercises**

1. **Experiment with Different Textures**: Try using different images as target textures and observe how the NCA learns to replicate them.

2. **Modify the Update Rules**: Adjust the neural network architecture or the perception kernels to see how it affects pattern formation.

3. **Implement 3D NCA**: Extend the model to 3D grids for volumetric texture generation.

4. **Combine Multiple NCAs**: Train multiple NCAs and blend their outputs for more intricate textures.

5. **Real-Time Interaction**: Create an interactive application where users can modify parameters on the fly.

---

## **Further Reading**

- **"Growing Neural Cellular Automata" by Mordvintsev et al.**: A seminal paper introducing NCAs for image generation.

- **GitHub Repositories**:
  - [PyTorch NCA Implementation](https://github.com/bertiebaggio/nca-pytorch)

- **Blog Posts**:
  - [Distill.pub Article on NCAs](https://distill.pub/2020/growing-ca/)

---

By incorporating Neural Cellular Automata into your programming toolkit, you can unlock new possibilities in procedural generation, creating textures and patterns that are both complex and organically evolving.

---

**Note**: Ensure that you have all the necessary libraries installed and that you are running the code in an environment that supports interactive plotting, such as Jupyter Notebook.