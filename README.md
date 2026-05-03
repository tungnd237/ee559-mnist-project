# EE559 Final Project — Handwritten Digit Classification

**Course:** EE559 · USC  
**Author:** Tung Nguyen · [tungdngu@usc.edu](mailto:tungdngu@usc.edu)

## Overview

Comparison of three classifiers on the MNIST handwritten digit dataset (0–9):

| Model | Test Accuracy | Weighted F1 |
|-------|:------------:|:-----------:|
| k-NN (k=7) | 96.11% | 96.09% |
| SVM RBF (C=1) | 97.78% | 97.77% |
| **MLP (4-layer)** | **98.33%** | **98.34%** |

## Dataset

The notebook uses **sklearn's `load_digits`** (1 797 samples, 8×8 images, 64 features) as a lightweight proxy for full MNIST. To switch to real MNIST (70 000 samples, 28×28), replace the loading block with:

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data.astype(np.float32), mnist.target.astype(int)
```

**Split:** 70% train / 10% validation / 20% test (stratified).

## Methods

### Preprocessing
- `StandardScaler` (zero mean, unit variance)
- PCA dimensionality reduction — 39 components retained (95% variance threshold) for k-NN and SVM

### Models
- **k-NN** — Euclidean distance; k swept over {1, 3, 5, 7, 9, 11, 15}; best k=7
- **SVM RBF** — `gamma='scale'`; C swept over {0.1, 1, 10, 50}; best C=1
- **MLP (PyTorch)** — 4 fully-connected layers with BatchNorm, ReLU, Dropout; Adam optimizer; early stopping (patience=10)

  Architecture for 64-dim input: `64 → 256 → 128 → 64 → 10`

## Repository Structure

```
ee559-mnist-project/
├── Final_project.ipynb        # Main notebook (data, training, evaluation, figures)
└── confusion_matrix_mnist.png # Confusion matrix visualization
```

## Requirements

```
numpy
matplotlib
scikit-learn
torch
```

Install with:

```bash
pip install numpy matplotlib scikit-learn torch
```

## Running

Open `Final_project.ipynb` in Jupyter or Google Colab and run all cells. Figures are saved to `./figs/` and results to `final_results.npy`.

> **Note:** The notebook was developed on Google Colab; paths like `/content/figs/` are Colab defaults. Change them to local paths if running locally.
