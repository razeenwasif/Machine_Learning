from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn 


# This file contains utility functions for training and visualizing the models in lab 02.


def visualize_loss_curve(losses: list[float], title: str = None):
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if title:
        plt.title(title)
    plt.show()


def visualize_decision_boundary(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    title: str = None,
):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color="red", label="train 0")
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color="blue", label="train 1")
    plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color="red", label="test 0", marker="x")
    plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color="blue", label="test 1", marker="x")
    x1 = torch.linspace(-0.5, 0.5, 100)
    x2 = torch.linspace(-0.5, 0.5, 100)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    X = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
    Y = model(X).argmax(dim=1)
    plt.contourf(X1, X2, Y.reshape(100, 100), alpha=0.3, colors=["red", "blue"], levels=1)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.legend()
    plt.show()


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    loss_fn: Callable,
    accuracy_fn: Callable,
    num_epochs: int = 10,
    print_every_epochs: int = 1,
    evaluate_every_epochs: int = 10,
    lr: float = 0.1,
    temperature: float | None = None,
):
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        if temperature:
            loss = loss_fn(y_pred, y_train, temperature)
        else:
            loss = loss_fn(y_pred, y_train)
        if (epoch + 1) % print_every_epochs == 0:
            print(f"Epoch: {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % evaluate_every_epochs == 0:
            with torch.no_grad():
                # calculate train accuracy and test accuracy
                y_pred_train = model(x_train)
                train_acc = accuracy_fn(y_pred_train, y_train)
                y_pred_test = model(x_test)
                test_acc = accuracy_fn(y_pred_test, y_test)
                # Visualize the decision boundary
                visualize_decision_boundary(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    f"Epoch: {epoch+1}/{num_epochs} Train Acc: {train_acc:.2f} Test Acc: {test_acc:.2f}",
                )
    # visualize the loss curve
    visualize_loss_curve(losses, "Loss Curve")