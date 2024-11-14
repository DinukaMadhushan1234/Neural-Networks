# Neural Networks
This repository contains my solutions for the EN3160 Assignment 3 on Neural Networks. The assignment includes three tasks:

1. Modifying a single dense layer network.
2. Implementing a LeNet-5 network for MNIST.
3. Using a pre-trained ResNet18 for transfer learning on the hymenoptera dataset.

## Contents
- `210349N_a03.ipynb`: Jupyter Notebook with the implemented solutions for the tasks.
- **Results**: Training and test accuracies, loss plots, and discussions on results.

---

## Tasks Overview

### Task 1: Modifying a Single Dense Layer Network

- **Objective:**
  - Add a middle layer with 100 nodes and a sigmoid activation.
  - Use cross-entropy loss.
  - Train the network for 10 epochs and report the training and test accuracies.

- **Highlights:**
  - Input: CIFAR-10 dataset.
  - Modified architecture includes an additional dense layer with activation.
  - Training and test accuracies reported.

### Task 2: LeNet-5 Implementation for MNIST

- **Objective:**
  - Implement the LeNet-5 convolutional neural network architecture using PyTorch.
  - Train the network on the MNIST dataset for 10 epochs.
  - Report training and test accuracies.

- **Highlights:**
  - Used PyTorch to define and train the LeNet-5 architecture.
  - MNIST dataset preprocessed using normalization and tensor conversion.

### Task 3: Transfer Learning with ResNet18

- **Objective:**
  - Classify the hymenoptera dataset using ResNet18 pre-trained on ImageNet.
  - Perform two approaches:
    1. Fine-tuning the model.
    2. Using the network as a feature extractor.
  - Report results.

- **Highlights:**
  - Fine-tuning and feature extraction approaches compared.
  - Training and test accuracies reported for both methods.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone /DinukaMadhushan1234/Neural-Networks
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook 210349N_a03.ipynb
   ```

---

## Notes

- The code and results are discussed comprehensively in the notebook.
- Ensure `torch`, `torchvision`, and other dependencies are installed to run the code.

---

### References

- [PyTorch Documentation](https://pytorch.org/)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

