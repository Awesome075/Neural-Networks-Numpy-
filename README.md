
# Neural Networks in Numpy

This project is an implementation of a neural network from scratch. The goal is to build a functional neural network library using only NumPy, for educational purposes.

---

## Current Features:
*   **Dense Layer**: A fully connected layer with forward and backward propagation.
*   **Activation Functions**: ReLU, Sigmoid, Softmax, and Linear activations with backward pass.
*   **Loss Functions**: MSE, MAE, SSE, Categorical Cross-Entropy, and Sparse Cross-Entropy.
*   **Backpropagation**: Full implementation of the backward pass.
*   **Optimizers**: An optimizer base class with an initial SGD (Stochastic Gradient Descent) implementation.
*	**Model API**: A Model class to easily build, compile, and train neural networks.
*	**He initialization**: Improve weight initialization for deeper networks.

---

## Future Goals:
*	**Advanced Optimizers**: Implement more sophisticated optimizers like Adam and RMSprop.
*	**Batch & Data Handling**: Support for mini-batch training and data shuffling for more efficient training on large datasets.
* 	**Metrics & Callbacks**: Include accuracy metrics during training and a callback system for actions like early stopping.
* 	**Advanced Layer Types**: Add more complex layers, such as Convolutional and Pooling layers.
* 	**Miscellaneous**: Code refactoring and improvements, better data handling and easier compatibility with pandas

---

## Setup Instructions

1. **Clone the Repository:**
```bash
git clone https://github.com/Awesome075/Neural-Networks-Numpy-.git
cd Neural-Networks-Numpy-
```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the Environment**
	
	- *On Windows:*
		```bash
		venv\Scripts\activate
		```

	- *On Linux/macOS:*
		```bash
		source venv/bin/activate
		```

4. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

5. **Run the Main example:**
```bash
python main.py
```