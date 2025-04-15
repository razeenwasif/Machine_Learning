# EMNIST Character Recognition using CNN and Optuna

This project implements a Convolutional Neural Network (CNN) model to recognize characters from the EMNIST dataset, which is an extended version of MNIST that includes handwritten digits and letters. The model is trained using PyTorch and optimized through hyperparameter tuning with Optuna.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- optuna
- numpy
- random

You can install the necessary dependencies using the following command:

```bash
pip install torch torchvision matplotlib optuna numpy
```

Dataset

The project uses the EMNIST dataset, which contains 814,255 characters across 814,255 images, representing digits and letters. The dataset is divided into training and testing sets. The balanced split of EMNIST is used, which includes characters from 0-9, A-Z, and a-z.

Usage
Training the Model

1. Clone this repository to your local machine:

2. Run the main.py script to train the model

This will download the EMNIST dataset, preprocess the images, and train the model. After training, the model will be saved as character_recognition.torch.

Hyperparameter Tuning with Optuna

The script includes a section for hyperparameter optimization using Optuna. The hyperparameters being tuned include:

    Learning Rate (lr)
    Dropout Rate (dropout_rate)
    Number of neurons in the first fully connected layer (fc1_neurons)
    Batch Size (batch_size)

The optimization process is currently commented out in the script. If you'd like to enable it, uncomment the section and set the number of trials for Optuna's optimization.

Making Predictions

Once the model is trained, you can use the predict_images() function to make predictions on random test images. The function will output the predicted character along with the actual label for a specified number of random test images.

Model Architecture

The model uses a simple CNN architecture with:

    Two convolutional layers with ReLU activations and max-pooling
    A dropout layer after the second convolutional layer
    Two fully connected layers for final classification
    Softmax activation at the output layer to produce class probabilities

The model uses the Adam optimizer and CrossEntropyLoss for training.
Hyperparameters

The following are the hyperparameters used for training the model:

    Learning Rate: 0.00047 (fixed)
    Dropout Rate: 0.34 (fixed)
    Number of Neurons in FC1 Layer: 196 (fixed)
    Batch Size: 256 (fixed)

These values were obtained through manual tuning and can be optimized further using Optuna.
Saving and Loading the Model

After training, the model is saved using torch.save()

To load the saved model for inference, uncomment the following line in the script:
# model.load_state_dict(torch.load('./character_recognition.torch'))

# Final model metrics:
Test loss: 0.0033
Accuracy: 86.08%
F1 score: 0.8592
Precision: 0.8636

Acknowledgments

    The EMNIST dataset is a part of the larger MNIST dataset collection and was developed by the Yale University School of Engineering and Applied Science.
    The PyTorch and Optuna libraries are used for building the neural network and hyperparameter tuning, respectively.
