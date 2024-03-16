# Convolutional Neural Network, CIFAR-10

In this notebook, I will train a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 database. The CIFAR-10 dataset consists of small color images grouped into ten classes, including objects like airplanes, automobiles, birds, cats, and more.

## Project Overview

The process will be broken down into the following steps:
### 1. Load Libraries
   - Importing essential libraries including PyTorch, NumPy, and Matplotlib for data manipulation, machine learning tasks, and visualization.

### 2. Load and Visualize the Data
   - Utilizing PyTorch to download and preprocess the CIFAR-10 dataset.
   - Splitting the dataset into training and testing sets, with a portion allocated for validation.
   - Creating data loaders for efficient loading of batches during training, validation, and testing.

### 3. Visualize a Batch of Training Data
   - Visualizing a batch of training images to gain insights into the nature of the data.
   - Displaying 20 images at a time, each labeled with its respective class.

### 4. View an Image in More Detail
   - Selecting and visualizing a single image in detail by extracting the RGB channels.
   - Creating subplots for each channel and annotating pixel values to understand their contribution to the overall color.

### 5. Define the Network Architecture
   - Defining a CNN architecture named "Net" for image classification.
   - The architecture includes convolutional layers, max-pooling layers, fully connected layers, and dropout layers for regularization.

### 6. Specify Loss Function and Optimizer
   - Loading necessary libraries and specifying categorical cross-entropy as the loss function.
   - Checking for GPU availability and moving the model to GPU if possible.
   - Choosing stochastic gradient descent (SGD) as the optimizer with a learning rate of 0.01.

### 7. Train the Network
   - Training the CNN on the CIFAR-10 dataset for 35 epochs.
   - Monitoring and printing training and validation losses after each epoch.
   - Saving the model state if the validation loss decreases, ensuring the best-performing model is retained.

### 8. Load the Model with the Lowest Validation Loss
   - Loading the model state from the file 'model_cifar.pt,' representing the model with the lowest validation loss.

### 9. Test the Trained Network
   - Evaluating the performance of the trained CNN on the test set.
   - Calculating and printing the average test loss, test accuracy for each class, and overall test accuracy.

### 10. Visualize Sample Test Results
   - Visualizing a batch of test images along with their predicted and true labels.
   - Displaying images in a grid with color-coded titles (green for correct predictions, red for incorrect), providing a quick overview of the model's performance.