![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Numpy%20Implementation-orange)
![Neural Network](https://img.shields.io/badge/Model-Type%3A%20L--Layer%20DNN-blueviolet)
![Jupyter Notebook](https://img.shields.io/badge/Made%20With-Jupyter%20Notebook-orange.svg)

# Deep Neural Network for Image Classification - Cat vs Non-Cat
This project implements a deep L-layer neural network from scratch using NumPy to classify images as cats or non-cats. The neural network is trained on a dataset of labeled cat and non-cat images, and achieves high accuracy on the training and test sets.

---

## Features
- Implements forward and backward propagation manually using NumPy
- Custom implementation of:
  - Linear and activation layers (ReLU, Sigmoid)
  - Cost computation using cross-entropy
  - Gradient descent optimization
- Vectorized implementation for speed and scalability
- Visualization of misclassified test examples
- Trains a 4-layer deep neural network: [LINEAR -> RELU] * 3 -> LINEAR -> SIGMOID

---

## Dataset
The dataset consists of:
- 209 training examples
- 50 testing examples
- Each image is of size 64x64x3 (RGB)
Dataset is loaded using a helper function from dnn_app_utils_v3.py.

---

## Model Architecture
```bash
[Input (12288)] → [Linear] → [ReLU]
                  ↓
                 [Linear] → [ReLU]
                  ↓
                 [Linear] → [ReLU]
                  ↓
                 [Linear] → [Sigmoid] → [Output (1)]
```

Layer dimensions:
```bash
layers_dims = [12288, 20, 7, 5, 1]
```

---

## Training Results
Training Accuracy: ~98.56%
Test Accuracy: ~80.00%

Achieves strong generalization and detects most cats and non-cats correctly.

---

## Installation & Running
1. Clone the repository
```bash
git clone https://github.com/your-username/cat-classifier-dnn.git
cd cat-classifier-dnn
```

2. Install Dependencies
```bash
pip install numpy matplotlib scipy pillow h5py
```

3. Run the Notebook
Make sure dnn_app_utils_v3.py is in the same directory as the notebook/script.
```bash
jupyter notebook cat_classifier.ipynb
```

---

## Results Visualization
View misclassified examples to better understand where the model may struggle:
```bash
print_mislabeled_images(classes, test_x, test_y, pred_test)
```
---

## Key Functions
| Function                       | Description                                      |
| ------------------------------ | ------------------------------------------------ |
| `initialize_parameters_deep()` | Initializes weights and biases for each layer    |
| `L_model_forward()`            | Implements forward propagation                   |
| `compute_cost()`               | Computes the binary cross-entropy cost           |
| `L_model_backward()`           | Implements backward propagation                  |
| `update_parameters()`          | Applies gradient descent to update parameters    |
| `predict()`                    | Predicts using learned parameters                |
| `print_mislabeled_images()`    | Displays images incorrectly labeled by the model |

---

## Training Logs
```bash
Cost after iteration 0: 0.7717493284237686
Cost after iteration 2499: 0.088439943441702

Train Accuracy: 98.56%
Test Accuracy: 80.00%
```

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
