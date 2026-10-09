# MNIST Handwritten-Digit Classifier with PyTorch

## Project objective

Build a handwritten-digit classifier using the MNIST dataset. The project first creates a simple linear baseline and then improves it with a small neural network.

### Required deliverables

- Working MNIST classifier
- Simple baseline
- Small neural network
- Five prediction examples
- Simple architecture sketch
- Explanation of input pixels → output class

## Learning resources

The project follows the beginner workflow from the official PyTorch "Learn the Basics" material: tensors, datasets/data loaders, transforms, model building, automatic differentiation, optimization, and saving/loading. The project also uses the introductory computer-vision ideas covered by Kaggle's Computer Vision course.

## 1. What is MNIST?

MNIST contains grayscale images of handwritten digits from 0 to 9.

Each image is:

- 28 × 28 pixels
- grayscale
- represented by numbers between 0 and 1 after `ToTensor()`
- assigned a label from 0 to 9

Therefore:

28 × 28 = 784 input pixel values.

## 2. Baseline

The baseline is a linear classifier:

784 pixels → Linear Layer → 10 output scores

It has no hidden layer and no nonlinear activation.

This gives us a simple model to compare against.

## 3. Neural-network model

The final model is:

784 → 128 → ReLU → 64 → ReLU → 10

The two hidden layers allow the model to learn more complex patterns in handwritten digits.

## 4. How training works

For every batch:

1. The image is passed into the model.
2. The model produces 10 scores.
3. Cross-entropy loss compares those scores with the correct digit.
4. `loss.backward()` calculates gradients.
5. The optimizer updates the model weights.
6. The process repeats over many batches and epochs.

## 5. Input → output explanation

Suppose the input is an image of handwritten `7`.

The 28 × 28 image contains 784 pixel values.

`Flatten()` changes the image from:

`[1, 28, 28]`

to a vector containing 784 values.

The first linear layer transforms these 784 values into 128 learned features.

ReLU adds nonlinearity.

The second linear layer produces 64 features, followed by another ReLU.

The final layer produces 10 scores:

`[score_0, score_1, ..., score_9]`

The class with the highest score becomes the prediction.

If the largest score is at index 7, the model predicts digit `7`.

## 6. Architecture sketch

```text
             MNIST IMAGE
              28 x 28
                 |
                 v
              Flatten
             784 values
                 |
                 v
           Linear 784→128
                 |
                ReLU
                 |
                 v
            Linear 128→64
                 |
                ReLU
                 |
                 v
            Linear 64→10
                 |
                 v
       10 output class scores
                 |
                 v
        Highest score = digit
```

## 7. How to run

Install the packages:

```bash
pip install torch torchvision matplotlib
```

Then run:

```bash
python mnist_classifier.py
```

The first run downloads MNIST automatically.

The program will:

- load the training and test data
- train the linear baseline
- train the neural network
- print both accuracies
- print five sample predictions
- display the five prediction images
- save `five_predictions.png`
- save the trained model as `mnist_small_nn.pth`

## 8. Expected output

Your exact accuracy will depend on the run and hardware, but you should see output similar to:

```text
Using device: cpu
Training images: 60000
Test images: 10000
One image shape: torch.Size([1, 28, 28])

--- Training baseline ---
Baseline epoch 1/3 ...
Baseline epoch 2/3 ...
Baseline epoch 3/3 ...

--- Training neural network ---
NN epoch 1/5 ...
NN epoch 2/5 ...
...
NN epoch 5/5 ...

================ RESULTS ================
Linear baseline accuracy: ...
Neural network accuracy:  ...
==========================================

Example 1: True digit = ..., Predicted digit = ...
Example 2: True digit = ..., Predicted digit = ...
Example 3: True digit = ..., Predicted digit = ...
Example 4: True digit = ..., Predicted digit = ...
Example 5: True digit = ..., Predicted digit = ...
```

Do not copy the example accuracy values into your report. Use the values from your own run.

## 9. Why the neural network is better than the baseline

The linear baseline can learn a direct relationship between the 784 pixels and the 10 classes.

The neural network has hidden layers and ReLU activations. This gives it more capacity to learn combinations and patterns in the pixels, so it can generally achieve higher accuracy.

## 10. Important PyTorch concepts demonstrated

### Tensor
A tensor stores the image data numerically.

### Dataset
The MNIST dataset stores images and their labels.

### DataLoader
The DataLoader provides data in batches and shuffles the training data.

### Model
`nn.Module` defines the neural network.

### Loss function
`CrossEntropyLoss` measures how different the prediction is from the correct class.

### Optimizer
Adam updates the network weights using the gradients.

### Autograd
PyTorch automatically calculates gradients during `loss.backward()`.

### Prediction
`argmax(dim=1)` selects the output class with the largest score.

## 11. Five prediction examples

The program generates the five examples automatically and saves them in:

`five_predictions.png`

For your final submission, include that generated image rather than inventing prediction results.

## 12. Project conclusion

This project demonstrates the complete beginner computer-vision workflow:

**Image → Tensor → Neural Network → Loss → Backpropagation → Optimization → Prediction**

The important idea is that the neural network does not receive the digit as a symbol such as "7". It receives numerical pixel values and learns patterns from training examples that allow it to classify new images.
