# Neural-Networks-with-PyTorch

# Neural Networks with PyTorch: Tensors and Datasets

Welcome to the **Neural Networks with PyTorch: Tensors and Datasets** repository!  
This project is a hands-on introduction to core PyTorch concepts, focusing on manipulating tensors, creating and transforming datasets, and understanding the foundational mechanics behind neural networks in PyTorch. Each Jupyter Notebook in this repository is a step-by-step learning resource, complete with code, explanations, and practical exercises.

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository is designed for learners who want a practical, example-driven approach to PyTorch. The notebooks cover:

- The basics of 1D and 2D tensors in PyTorch and related data types
- Indexing, slicing, and manipulating tensors
- Building and transforming custom datasets
- Using pre-built datasets and torchvision transforms
- Understanding PyTorch’s autograd system for derivatives and computational graphs

Whether you are a beginner or refreshing your PyTorch skills, these notebooks will help you build a strong foundation for deep learning projects.

---

## Notebooks

### 1. **1D tensors.ipynb**
Learn the basics of 1-dimensional tensors in PyTorch:
- Creating tensors from lists, numpy arrays, and pandas Series
- Type conversion (int, float, etc.) and reshaping with `.view()`
- Indexing, slicing, and conversion between tensor and list/numpy
- Practical tips for working with 1D data

### 2. **Two-Dimensional Tensors.ipynb**
Dive into 2D tensors (matrices):
- Creating and manipulating 2D tensors
- Shape, dimension, and element count operations
- Conversion to/from numpy arrays and pandas DataFrames
- Indexing, slicing, and common tensor operations (addition, multiplication, matrix multiplication)

### 3. **simple data set.ipynb**
Build your own custom dataset:
- Creating a PyTorch `Dataset` class from scratch
- Custom transforms and composing transforms
- Applying and chaining multiple transformations
- Iterating over datasets and understanding dataset length and indexing

### 4. **Datasets and transforms.ipynb**
Work with real-world datasets and data preprocessing:
- Downloading and unpacking the FashionMNIST dataset
- Loading data into pandas DataFrames
- Mapping numerical labels to descriptive categories
- Inspecting and visualizing dataset structure

### 5. **Pre-Built Datasets and transforms.ipynb**
Explore torchvision datasets and image transformations:
- Using prebuilt datasets like MNIST
- Applying and composing image transforms (crop, flip, etc.)
- Visualizing transformed images
- Practical usage of `torchvision.transforms.Compose`

### 6. **derivatives and Graphs in Pytorch.ipynb**
Understand PyTorch’s autograd and computational graphs:
- Computing derivatives and gradients with `requires_grad`
- Understanding leaf tensors, backward passes, and gradients
- Custom autograd functions
- Visualizing functions and their derivatives
- Partial derivatives and working with functions of multiple variables

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/usafhulk/Neural-Networks-with-PyTorch-Tensor-and-Datasets.git
   cd Neural-Networks-with-PyTorch-Tensor-and-Datasets
Install dependencies:

Python 3.7 or newer
PyTorch
torchvision
numpy, pandas, matplotlib, jupyter
Install with pip:

bash
pip install torch torchvision numpy pandas matplotlib jupyter
Run the notebooks:

bash
jupyter notebook
Open the notebook you want to explore and follow the instructions inside.

Contributing
Contributions, suggestions, and corrections are welcome!
Feel fre to open issues or pull requests to improve the materials or add new learning resources.


Happy Learning!
Feel free to reach out with questions or ideas for further notebooks.
