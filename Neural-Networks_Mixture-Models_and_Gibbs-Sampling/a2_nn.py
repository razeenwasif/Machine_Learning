import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Relu():
    ##############
    #Question 2.1#
    ##############

    def __init__(self):
        ################
        #YOUR CODE HERE#
        ################
        self.input_x = None 

    def forward_pass(self, X):
        ################
        #YOUR CODE HERE#
        ################
        self.input_x = X 
        return np.maximum(0, X)

    def backward_pass(self, g_next_layer):
        ################
        #YOUR CODE HERE#
        ################
        if self.input_x is None:
            raise ValueError("Forward pass must be called before backward_pass")
        return g_next_layer * (self.input_x > 0)

    def update(self, learning_rate):
        ################
        #YOUR CODE HERE#
        ################
        pass


class SquaredErrorLoss():

    ##############
    #Question 2.2#
    ##############

    def __init__(self):
        ################
        #YOUR CODE HERE#
        ################
        self.y_pred = None 
        self.t_true = None 
        self.N = None 

    def forward_pass(self, y, t):
        ################
        #YOUR CODE HERE#
        ################
        self.y_pred = y 
        self.t_true = t

        # ensure y_pred and t_true are 2D arr (N, 1)
        if self.y_pred.ndim == 1:
            self.y_pred = self.y_pred.reshape(-1, 1)
        if self.t_true.ndim == 1:
            self.t_true = self.t_true.reshape(-1, 1)

        if self.y_pred.shape[0] != self.t_true.shape[0]:
            raise ValueError("Prediction and target must have the same number of samples.")

        self.N = self.y_pred.shape[0]
        if self.N == 0:
            return 0.0 
        
        loss_val = (1.0 / self.N) * np.sum((self.t_true - self.y_pred)**2)
        return loss_val

    def backward_pass(self, g_next_layer = 1):
        ################
        #YOUR CODE HERE#
        ################
        if self.y_pred is None or self.t_true is None or self.N is None or self.N == 0:
            # Forward pass was not called or an empty batch was processed 
            # return an zero grad 
            if self.y_pred is not None:
                return np.zeros_like(self.y_pred)
            # cannot determine shape if y_pred is None 
            raise ValueError("Forward pass must be called with valid data before backward pass")

        grad = (2.0 / self.N) * (self.y_pred - self.t_true)
        return grad * g_next_layer

    def update(self, learning_rate):
        ################
        #YOUR CODE HERE#
        ################
        pass


class Network():

    ##############
    #Question 2.3#
    ##############
    # Input layer: 1 node 
    # Hidden layer 1: P nodes, ReLU 
    # Hidden layer 2: Q nodes, ReLU 
    # Hidden layer 3: 2 nodes, ReLU 
    # Output player: 1 node, no activation 

    def __init__(self, P, Q):
        ################
        #YOUR CODE HERE#
        ################ 

        # Layer 1 weights and biases 
        self.w1 = np.random.randn(1, P) * np.sqrt(2/1)
        self.b1 = 0.01 * np.random.randn(P)
        self.relu1 = Relu()

        # Layer 2 weights and biases 
        self.w2 = np.random.randn(P, Q) * np.sqrt(2/P)
        self.b2 = 0.01 * np.random.randn(Q)
        self.relu2 = Relu()

        # Layer 3 weights and biases 
        self.w3 = np.random.randn(Q, 2) * np.sqrt(2/Q)
        self.b3 = 0.01 * np.random.randn(2)
        self.relu3 = Relu()

        # Output Layer weights and biases 
        self.w4 = np.random.randn(2, 1) * np.sqrt(2/2)
        self.b4 = 0.01 * np.random.randn(1)

        # store intermediate values for backward pass 
        self.x0 = None
        self.a1, self.x1 = None, None
        self.a2, self.x2 = None, None 
        self.a3, self.x3 = None, None 
        self.a4 = None 

        # store gradients 
        self.dw1, self.db1 = None, None
        self.dw2, self.db2 = None, None
        self.dw3, self.db3 = None, None
        self.dw4, self.db4 = None, None

        
    def forward_pass(self, X):
        ################
        #YOUR CODE HERE#
        ################
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # input to layer 1 
        self.x0 = X 

        # Layer 1: Linear -> Relu 
        self.a1 = self.x0 @ self.w1 + self.b1 
        self.x1 = self.relu1.forward_pass(self.a1) # output of relu1
        
        # Layer 2: Linear -> ReLU 
        self.a2 = self.x1 @ self.w2 + self.b2 
        self.x2 = self.relu2.forward_pass(self.a2) # output of relu2

        # Layer 3: Linear -> ReLU 
        self.a3 = self.x2 @ self.w3 + self.b3 
        self.x3 = self.relu3.forward_pass(self.a3) # output of relu3

        # Layer 4 (output): Linear
        self.a4 = self.x3 @ self.w4 + self.b4 
        self.y_pred = self.a4 # final output 

        return self.y_pred

    def backward_pass(self, grad):
        ################
        #YOUR CODE HERE#
        ################
        grad_a4 = grad 

        # Layer 4 (Linear: w4, b4)
        # dL/dw4 = dL/da4 * da4/dw4 = dL/da4 * x3.T 
        self.dw4 = self.x3.T @ grad_a4 
        # dL/db4 = dL/da4 * da4/db4 = dL/da4 * 1 
        self.db4 = np.sum(grad_a4, axis=0) 
        # dL/dx3 = dL/da4 * da4/dx3 = dL/da4 * w4.T 
        grad_x3 = grad_a4 @ self.w4.T 

        # Relu 3 backward pass 
        # dL/da3 = dL/dx3 * dx3/da3 
        grad_a3 = self.relu3.backward_pass(grad_x3)

        # Layer 3 (Linear: w3, b3)
        self.dw3 = self.x2.T @ grad_a3 
        self.db3 = np.sum(grad_a3, axis=0)
        grad_x2 = grad_a3 @ self.w3.T 

        # Relu 2 backward pass 
        grad_a2 = self.relu2.backward_pass(grad_x2)

        # Layer 2 (Linear: w2, b2)
        self.dw2 = self.x1.T @ grad_a2 
        self.db2 = np.sum(grad_a2, axis=0)
        grad_x1 = grad_a2 @ self.w2.T 

        # Relu 1 backward pass 
        grad_a1 = self.relu1.backward_pass(grad_x1)

        # Layer 1 (Linear: w1, b1)
        self.dw1 = self.x0.T @ grad_a1 
        self.db1 = np.sum(grad_a1, axis=0)

    def update(self, learning_rate):
        ################
        #YOUR CODE HERE#
        ################
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1
        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2
        self.w3 -= learning_rate * self.dw3
        self.b3 -= learning_rate * self.db3
        self.w4 -= learning_rate * self.dw4
        self.b4 -= learning_rate * self.db4

# The following code is provided and should not be changed.
# The code prepares the training and test data for the two regression tasks.
np.random.seed(1)

# Generate data points for the regression problem
# Generate 10,000 random numbers inside [-5, 5]
x = np.random.random((10000, 1)) * 10 - 5

# choose regression task and P, Q values 
task_name = "x_abs" # x_squared or x_abs 
P_conf = 10 # 2, 5, 10
Q_conf = 10 # 2, 5, 10

print(f"--- Task to run: {task_name}, P={P_conf}, Q={Q_conf} ---")

# Generate y
if task_name == "x_squared":
    # generate y = x^2
    y = x ** 2
elif task_name == "x_abs":
    # generate y = abs(x)
    y = np.abs(x)
else:
    raise ValueError("task_name must be 'x_squared' or 'x_abs'")

y = y.reshape(-1)


# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# Set the number of epochs
num_training_epochs = 1000

# Set P and Q, the numbers of hidden nodes
# Modify the code below to configure the network differently
P = P_conf
Q = Q_conf

# Set the learning rate
learning_rate = 0.001

# Create the network
np.random.seed(1)
network = Network(P, Q)

################
#YOUR CODE HERE#
################

# Write your code that trains a neural network using (x_train, y_train) and test it on (x_test, y_test)
loss_fn = SquaredErrorLoss()
train_losses = []
test_losses = [] 

for epoch in range(num_training_epochs):
    # Training phase 
    y_pred_train = network.forward_pass(x_train)
    current_train_loss = loss_fn.forward_pass(y_pred_train, y_train)
    train_losses.append(current_train_loss)

    # Backward pass and update 
    grad_from_loss = loss_fn.backward_pass()
    network.backward_pass(grad_from_loss)
    network.update(learning_rate)

    # Testing phase 
    y_pred_test = network.forward_pass(x_test)
    current_test_loss = loss_fn.forward_pass(y_pred_test, y_test)
    test_losses.append(current_test_loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_training_epochs}, Train loss: {current_train_loss:.6f}, Test loss: {current_test_loss:.6f}")

# Plot 1: Training loss and test loss over epochs 
plt.figure(figsize=(10, 6))
plt.plot(range(num_training_epochs), train_losses, label='Training Loss')
plt.plot(range(num_training_epochs), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel("Loss (Squred Error)")
plt.title(f"Training and Test Loss\nTask: {task_name}, P={P}, Q={Q}, LR={learning_rate}")
plt.legend()
plt.grid(True)
plt.savefig(f"loss_plot_{task_name}_P{P}_Q{Q}.png")
plt.show()

# Plot 2: Actual output value of y and predicted output value of y over x_test 
y_pred_final_on_test = network.forward_pass(x_test)
# sort for smooth line plot 
sort_indices = np.argsort(x_test[:, 0])
x_test_sorted = x_test[sort_indices]
y_test_sorted = y_test[sort_indices]
y_pred_final_on_test_sorted = y_pred_final_on_test[sort_indices]

plt.figure(figsize=(10,6))
plt.scatter(x_test, y_test, label="Actual y_test (Data Points)", color="blue", alpha=0.3, s=10)
plt.plot(x_test_sorted, y_test_sorted, label="Actual y_test (True Function)", color="blue", linestyle='--')
plt.plot(x_test_sorted, y_pred_final_on_test_sorted, label='Predicted y_test (Network Output)', color='red', linewidth=2)
plt.xlabel("x_test")
plt.ylabel("y")
plt.title(f'Actual vs. Predicted y on Test Data\nTask: {task_name}, P={P}, Q={Q}')
plt.legend()
plt.grid(True)
plt.savefig(f'fit_plot_{task_name}_P{P}_Q{Q}.png')
plt.show()

############## Question 2.5 ##################

if task_name == "x_abs":
    _ = network.forward_pass(x_test_sorted)
    z3_node1_outputs = network.x3[:, 0]
    z3_node2_outputs = network.x3[:, 1]

    plt.figure(figsize=(12,7))
    plt.plot(x_test_sorted, z3_node1_outputs, label="z(3)_1 Output")
    plt.plot(x_test_sorted, z3_node2_outputs, label="z(3)_2 Output")
    plt.plot(x_test_sorted, np.abs(x_test_sorted), label='True abs(x) for reference', color='k', linestyle=':')
    plt.xlabel("x_test")
    plt.ylabel("Activation Value")
    plt.title(f"Last Hidden Layer Node Outputs ({task_name}, P={P}, Q={Q})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"q2_5_hidden_nodes_{task_name}_P{P}_Q{Q}.png")
    plt.show() 

    # Print out weights connecting z(3) to y 
    print("\nActual weights from the last hidden layer (z(3)) to the output node y:")
    print(f"Weight for z(3)_1 -> y: {network.w4[0, 0]}")
    print(f"Weight for z(3)_2 -> y: {network.w4[1, 0]}")
    print(f"Bias of the output node y (b4): {network.b4[0]:.4f}")
