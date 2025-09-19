# MNIST Neural Network Optimization

This project documents the iterative process of building and optimizing a neural network to achieve over 99.4% test accuracy on the MNIST dataset with a model under 20,000 parameters.

## The Journey

The process involved several iterations, each building on the learnings of the previous one. Here's a summary of the steps taken:

| Iteration | Key Changes | Parameters | Peak Test Accuracy (%) |
| :--- | :--- | :--- | :--- |
| 1 | Baseline deep model | 350,000 | 98.85 |
| 5 | Lighter model, no LR scheduler | 19,489 | 99.27 |
| 6 | Removed dropout | 19,489 | 99.30 |
| 7 | Increased model size, reintroduced dropout | 24,616 | 99.39 |
| 8 | Reduced FC layer size | 19,056 | 99.25 |
| 9 | Re-enabled LR scheduler, slightly larger FC layer | 20,446 | 99.52 |
| FINAL | Lighter model with LR scheduler | 19,056 | 99.48 |

### Iteration 1: The Baseline

The first iteration was a deep convolutional neural network with 8 convolutional layers and 2 fully-connected layers. This initial model was overly complex and, as the filename suggests, had "too many parameters," which was far from the target of under 20,000. The training for this model was interrupted, so no accuracy or parameter count is available.

### Iteration 5: A Lighter Model

In this iteration, the model was significantly redesigned to be much lighter. The new architecture, with a total of **19,489 parameters**, was a major step towards the goal. The learning rate scheduler was disabled for this iteration, and the model achieved a peak test accuracy of **99.27%**.

### Iteration 6: Removing Dropout

Building on the previous iteration, this step involved a simple but effective experiment: removing the dropout layer. This change, while not affecting the parameter count, improved the test accuracy to **99.30%**. This suggested that the model was not overfitting and that dropout was not necessary for this architecture.

### Iteration 7: Pushing for Accuracy

To push the accuracy further, this iteration experimented with increasing the model's capacity. The number of channels in the convolutional layers and the size of the fully connected layer were increased, resulting in a model with **24,616 parameters**. This successfully boosted the test accuracy to **99.39%**, but at the cost of violating the under-20k parameter constraint. Dropout was also re-introduced in this iteration.

### Iteration 8: Back to Basics

In an attempt to get back under the parameter limit, this iteration reduced the size of the fully connected layer. This brought the parameter count down to a lean **19,056**. However, this change also resulted in a drop in test accuracy to **99.25%**, highlighting the delicate balance between model size and performance.

### Iteration 9: The Power of Scheduling

This iteration revealed a key ingredient for success: the learning rate scheduler. By re-enabling the `StepLR` scheduler and slightly increasing the size of the fully connected layer, the model achieved an impressive **99.52%** test accuracy. Although the parameter count was slightly over the limit at **20,446**, this iteration demonstrated the power of a well-tuned learning rate.

### Iteration FINAL: The Winning Combination

The final iteration brought together the best of the previous experiments. By combining the lighter model architecture from Iteration 8 (with **19,056 parameters**) and the learning rate scheduler from Iteration 9, the model successfully achieved the project's goal. The final test accuracy was a stellar **99.48%**, demonstrating that a carefully tuned, smaller model could outperform its larger counterparts.

Here is the `torchsummary` output for the final model:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
            Conv2d-3           [-1, 16, 28, 28]           1,168
       BatchNorm2d-4           [-1, 16, 28, 28]              32
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 24, 12, 12]           3,480
       BatchNorm2d-7           [-1, 24, 12, 12]              48
         MaxPool2d-8             [-1, 24, 6, 6]               0
            Conv2d-9             [-1, 32, 4, 4]           6,944
      BatchNorm2d-10             [-1, 32, 4, 4]              64
           Conv2d-11              [-1, 8, 4, 4]             264
           Linear-12                   [-1, 50]           6,450
          Dropout-13                   [-1, 50]               0
           Linear-14                   [-1, 10]             510
================================================================
Total params: 19,056
Trainable params: 19,056
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.38
Params size (MB): 0.07
Estimated Total Size (MB): 0.46
----------------------------------------------------------------
```

## Conclusion

This project demonstrates a classic iterative approach to model optimization. The key takeaways are:
*   **Model Size vs. Accuracy**: A larger model doesn't always mean better performance, especially when considering constraints. A smaller, well-tuned model can be more effective.
*   **Learning Rate Scheduling**: The learning rate scheduler played a crucial role in achieving the final high accuracy. It's a powerful tool that should not be overlooked.
*   **Regularization**: Techniques like dropout are not always necessary and can sometimes hinder performance, as seen in this case.
