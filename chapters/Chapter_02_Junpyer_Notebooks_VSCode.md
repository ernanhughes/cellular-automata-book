# **Introduction to Jupyter Notebooks in Visual Studio Code**

Jupyter Notebooks are a powerful tool for interactive computing, widely used in data science, machine learning, and scientific research. While traditionally accessed through a web browser, you can enhance your productivity by using Jupyter Notebooks directly within Visual Studio Code (VSCode). This chapter will guide you through the fundamentals of using Jupyter Notebooks in VSCode, integrating them seamlessly into your Python development workflow.

---

# **Why Use Jupyter Notebooks in VSCode?**

Using Jupyter Notebooks within VSCode offers several advantages:

- **Integrated Development Environment**: Benefit from VSCode's powerful features like IntelliSense, debugging, and source control.
- **Single Interface**: Work on notebooks and traditional Python scripts in the same environment.
- **Extensions and Customization**: Leverage a wide range of VSCode extensions to enhance your workflow.
- **Performance**: Enjoy faster performance compared to the traditional web-based Jupyter Notebook interface.

---

# **Getting Started with VSCode and Jupyter Notebooks**

## **Prerequisites**

- **Visual Studio Code**: Download and install from [official website](https://code.visualstudio.com/).
- **Python**: Ensure Python is installed on your system. Download from [python.org](https://www.python.org/downloads/).
- **Jupyter Package**: Install Jupyter via `pip`.

---

## **Step 1: Install Visual Studio Code**

1. **Download VSCode**: Visit the [VSCode download page](https://code.visualstudio.com/download) and select the appropriate installer for your operating system.
2. **Install VSCode**: Run the installer and follow the on-screen instructions.

---

## **Step 2: Install the Python Extension**

1. **Open VSCode**.
2. **Access Extensions**: Click on the **Extensions** icon on the left sidebar or press `Ctrl + Shift + X`.
3. **Search for Python**: In the search bar, type **`Python`**.
4. **Install Python Extension**: Locate the official **Python** extension by Microsoft and click **Install**.

   ![Python Extension](attachment:python_extension.png)

---

## **Step 3: Install Jupyter Package**

You need the Jupyter package installed in your Python environment to use notebooks.

### **Using pip**

Open a terminal or command prompt and run:

```bash
pip install jupyter
```

### **Using Conda (Optional)**

If you're using Anaconda:

```bash
conda install jupyter
```

---

## **Step 4: Create a New Jupyter Notebook in VSCode**

1. **Open Command Palette**: Press `Ctrl + Shift + P` (Windows/Linux) or `Cmd + Shift + P` (macOS).
2. **Create New Notebook**: Type **`Jupyter: Create New Jupyter Notebook`** and select it.

   ![Create New Notebook](attachment:create_new_notebook.png)

3. **Select Kernel**: Choose the Python interpreter you want to use as the kernel.

   ![Select Kernel](attachment:select_kernel.png)

---

# **Understanding the Notebook Interface in VSCode**

When you create or open a notebook in VSCode, the interface consists of:

- **Notebook Toolbar**: Contains actions like running cells, adding cells, saving, etc.
- **Cells**: Interactive units where you write code or markdown.
- **Outline View**: Provides a structural view of the notebook.

---

## **Working with Cells**

### **Adding Cells**

- **Add Code Cell**: Click the **`+ Code`** button or hover between cells and click **`Add Code`**.
- **Add Markdown Cell**: Click the dropdown next to **`+ Code`** and select **`+ Markdown`**.

### **Editing Cells**

- **Edit Mode**: Click inside a cell to edit its contents.
- **Command Mode**: Press `Esc` to enable keyboard shortcuts for cell operations.

### **Running Cells**

- **Run Cell**: Click the **`Run`** button on the left of the cell or press `Shift + Enter`.
- **Run All Cells**: Click **`Run All`** in the notebook toolbar.

---

## **Example: Hello World**

### **Code Cell**

```python
print("Hello, World!")
```

**Execution:**

- Run the cell using the **`Run`** button or `Shift + Enter`.

**Output:**

```
Hello, World!
```

---

## **Using Markdown Cells**

Markdown cells allow you to include formatted text, images, and equations.

### **Creating a Markdown Cell**

1. **Add Markdown Cell**: Click **`+ Markdown`**.
2. **Edit Content**: Write your markdown-formatted text.
3. **Render Markdown**: Click **`Run`** or press `Shift + Enter` to render.

### **Example: Markdown Cell**

```markdown
# This is a Heading

This is **bold** text and this is *italic* text.

- Item 1
- Item 2
```

**Rendered Output:**

# This is a Heading

This is **bold** text and this is *italic* text.

- Item 1
- Item 2

---

# **Interactive Computing in VSCode Notebooks**

## **Importing Libraries**

You can import any Python library in a code cell.

```python
import numpy as np
import pandas as pd
```

---

## **Data Visualization**

VSCode notebooks support inline plotting with libraries like Matplotlib.

### **Example: Simple Plot**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output:**

![Sine Wave Plot](attachment:sine_wave_plot.png)

---

## **Interactive Widgets**

As of now, VSCode's support for interactive widgets like `ipywidgets` is limited. However, basic interactivity can be achieved.

### **Note**

- Some interactive features may not work as expected.
- Future updates may improve widget support.

---

# **Managing Notebooks in VSCode**

## **File Operations**

- **Save Notebook**: Click **`Save`** or press `Ctrl + S`.
- **Rename Notebook**: Right-click the notebook file in the Explorer and select **`Rename`**.
- **Close Notebook**: Click the **`X`** on the notebook tab.

---

## **Kernel Management**

- **Change Kernel**: Click on the kernel name in the upper right corner to switch.
- **Restart Kernel**: Click **`Restart`** in the notebook toolbar.

---

## **Notebook Outline**

- **Access Outline**: Click on the **`Outline`** view to see a structural overview.
- **Navigate**: Click on items in the outline to jump to that section.

---

# **Version Control with Git**

VSCode integrates seamlessly with Git, allowing you to version control your notebooks.

## **Initializing a Git Repository**

1. **Open Source Control**: Click on the **Source Control** icon or press `Ctrl + Shift + G`.
2. **Initialize Repository**: Click **`Initialize Repository`**.

## **Committing Changes**

- **Stage Changes**: Click **`+`** next to changed files.
- **Commit**: Enter a commit message and press **`Ctrl + Enter`**.

## **Viewing Diffs**

- VSCode can display diffs of notebook files in JSON format.
- For better notebook diffs, consider using external tools like **`nbdime`**.

---

# **Keyboard Shortcuts in VSCode Notebooks**

## **Common Shortcuts**

- **Run Cell**: `Shift + Enter`
- **Add Cell Below**: `Ctrl + Shift + B`
- **Delete Cell**: Select cell and press `DD` (double `D` in command mode)
- **Move Cell Up/Down**: `Alt + Up/Down Arrow`
- **Convert Cell to Code/Markdown**: `Ctrl + M` then `Y` (code) or `M` (markdown)

## **Viewing All Shortcuts**

- **Open Keyboard Shortcuts**: `Ctrl + K` then `Ctrl + S`
- **Filter for Notebook Shortcuts**: Type **`Notebook`** in the search bar.

---

# **Extensions and Customization**

## **Jupyter Extension**

- The **Jupyter** extension by Microsoft enhances notebook support.
- **Installation**: Should be installed automatically with the Python extension. If not, install it from the Extensions marketplace.

## **Customization Options**

- **Settings**: Adjust notebook settings via **`File > Preferences > Settings`**.
- **Themes**: Change the appearance using VSCode themes.

---

# **Exporting and Converting Notebooks**

## **Export Formats**

- **Python Script**: Convert notebook to a `.py` file.
- **HTML**: Export as an HTML file for sharing.
- **PDF**: Export as a PDF (requires additional setup).

## **Exporting Steps**

1. **Open Command Palette**: `Ctrl + Shift + P`.
2. **Export Notebook**: Type **`Export Current Notebook As`** and select the desired format.

   ![Export Notebook](attachment:export_notebook.png)

---

# **Debugging in Notebooks**

VSCode allows you to debug code within notebooks.

## **Setting Breakpoints**

- **Add Breakpoint**: Click in the gutter next to a code line.
- **Start Debugging**: Click **`Debug Cell`** or **`Run by Line`**.

## **Using the Debugger**

- **Step Over**: Execute the next line.
- **Variables View**: Inspect variables in the current scope.
- **Call Stack**: View the call stack to trace execution flow.

---

# **Best Practices**

## **Organize Your Notebook**

- Use headings and markdown cells to structure your content.
- Keep code cells concise and focused on single tasks.

## **Use Virtual Environments**

- Create a virtual environment for your project.
- Select the environment as your Python interpreter in VSCode.

## **Version Control**

- Regularly commit changes to your repository.
- Consider cleaning outputs before committing to reduce file size.

---

# **Advanced Features**

## **Using Interactive Python Scripts**

VSCode supports running Python scripts interactively.

### **Steps**

1. **Open a Python Script**: `.py` file.
2. **Add `# %%`**: Mark cells by adding `# %%` at the beginning of a line.
3. **Run Cells**: Click **`Run Cell`** links that appear above each cell.

---

## **Remote Development**

- Use VSCode's **Remote Development** extension to work on notebooks hosted on remote servers or containers.
- **SSH into Remote Machine**: Edit and run notebooks as if they were local.

---

# **Troubleshooting**

## **Common Issues**

### **Kernel Not Found**

- **Solution**: Ensure the correct Python interpreter is selected.
- **Check Kernel**: Click on the kernel name and select the appropriate environment.

### **Extensions Not Working**

- **Solution**: Update VSCode and extensions to the latest versions.

---

# **Conclusion**

Using Jupyter Notebooks within Visual Studio Code combines the best of both worlds: the interactive computing capabilities of notebooks and the powerful features of a modern IDE. This integration streamlines your workflow, allowing you to:

- **Develop Efficiently**: Write, run, and debug code in a single environment.
- **Stay Organized**: Manage notebooks alongside other project files.
- **Leverage Extensions**: Enhance functionality with a rich ecosystem of extensions.

---

# **Further Reading**

- **VSCode Documentation**: [Working with Jupyter Notebooks](https://code.visualstudio.com/docs/python/jupyter-support)
- **Python Extension**: [Python in VSCode](https://code.visualstudio.com/docs/languages/python)
- **Jupyter Extension**: [Jupyter in VSCode](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

---

# **Exercises**

1. **Create a Notebook**: Practice creating a new notebook in VSCode and executing simple code cells.
2. **Markdown Formatting**: Write a markdown cell using different formatting options.
3. **Data Visualization**: Import a dataset using Pandas and create a plot.
4. **Debugging**: Set breakpoints in your code and step through execution.
5. **Exporting**: Export your notebook as a Python script and HTML file.

