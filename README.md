# Convolutional Neural Networks (CNN) - Step by Step from Scratch

This project implements the foundational building blocks of **Convolutional Neural Networks (CNNs)** completely from scratch using **NumPy**, without relying on deep learning frameworks like TensorFlow or PyTorch.

##  Project Overview

The project covers both the Forward Pass and Backward Pass mechanisms of CNN architecture, implemented step by step:

1. **Zero Padding (`zero_pad`)**: Pads the borders of images with zeros to maintain spatial dimensions during convolution.
2. **Single Step Convolution (`conv_single_step`)**: Applies a filter to a single slice of the input to compute a single output neuron value.
3. **Convolution Forward Pass (`conv_forward`)**: Performs the full convolution operation across a batch of images, filters, and channels.
4. **Pooling Forward Pass (`pool_forward`)**: Implements max pooling and average pooling operations.
5. **Convolution Backward Pass (`conv_backward`)**: Computes gradients with respect to weights (`dW`), biases (`db`), and previous layer activations (`dA_prev`) based on cost gradients.
6. **Pooling Backward & Helpers (`create_mask_from_window`, `distribute_value`, `pool_backward`)**: Implements backward propagation mechanisms and helper functions for pooling layers.

---


##  Code Structure

* `zero_pad(X, pad)`: Pads the input tensor.
* `conv_single_step(a_slice_prev, W, b)`: Performs a single-window convolution step.
* `conv_forward(A_prev, W, b, hparameters)`: Executes the forward pass for a convolutional layer.
* `conv_backward(dZ, cache)`: Executes the backward pass (backpropagation) for a convolutional layer.
* `pool_forward(A_prev, hparameters, mode)`: Executes the forward pass for a pooling layer (`max` / `average`).
* `pool_backward(dA, cache, mode)`: Executes the backward pass for a pooling layer.

---

