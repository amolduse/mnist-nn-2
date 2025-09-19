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

The first iteration was a deep convolutional neural network with 8 convolutional layers and 2 fully-connected layers. This initial model was overly complex and, as the filename suggests, had "too many parameters," which was far from the target of under 20,000. Test accuracy was little under 99.

### Iteration 2 -4: Try to reduce parameters
In these iterations, I focused on reducing parameters and I did not maintain the source and logs. But the changes were very basic

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

Epoch 1
Train: Loss=0.1029 Batch_id=117 Accuracy=81.52: 100%|██████████| 118/118 [00:56<00:00,  2.09it/s]
Test set: Average loss: 0.0002, Accuracy: 9758/10000 (97.58%)

Epoch 2
Train: Loss=0.0835 Batch_id=117 Accuracy=96.79: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0001, Accuracy: 9855/10000 (98.55%)

Epoch 3
Train: Loss=0.0635 Batch_id=117 Accuracy=97.68: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0001, Accuracy: 9880/10000 (98.80%)

Epoch 4
Train: Loss=0.0566 Batch_id=117 Accuracy=98.08: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0001, Accuracy: 9905/10000 (99.05%)

Epoch 5
Train: Loss=0.1050 Batch_id=117 Accuracy=98.31: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0001, Accuracy: 9916/10000 (99.16%)

Epoch 6
Train: Loss=0.0677 Batch_id=117 Accuracy=98.42: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0001, Accuracy: 9918/10000 (99.18%)

Epoch 7
Train: Loss=0.0472 Batch_id=117 Accuracy=98.61: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0001, Accuracy: 9923/10000 (99.23%)

Epoch 8
Train: Loss=0.0395 Batch_id=117 Accuracy=98.72: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0000, Accuracy: 9926/10000 (99.26%)

Epoch 9
Train: Loss=0.0240 Batch_id=117 Accuracy=98.67: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0000, Accuracy: 9924/10000 (99.24%)

Epoch 10
Train: Loss=0.0432 Batch_id=117 Accuracy=98.79: 100%|██████████| 118/118 [00:55<00:00,  2.14it/s]
Test set: Average loss: 0.0000, Accuracy: 9924/10000 (99.24%)

Epoch 11
Train: Loss=0.0591 Batch_id=117 Accuracy=98.82: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0000, Accuracy: 9925/10000 (99.25%)

Epoch 12
Train: Loss=0.0709 Batch_id=117 Accuracy=98.95: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0001, Accuracy: 9927/10000 (99.27%)

Epoch 13
Train: Loss=0.0313 Batch_id=117 Accuracy=98.91: 100%|██████████| 118/118 [00:55<00:00,  2.14it/s]
Test set: Average loss: 0.0000, Accuracy: 9930/10000 (99.30%)

Epoch 14
Train: Loss=0.0848 Batch_id=117 Accuracy=98.93: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0000, Accuracy: 9928/10000 (99.28%)

Epoch 15
Train: Loss=0.0694 Batch_id=117 Accuracy=98.99: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0000, Accuracy: 9942/10000 (99.42%)

Epoch 16
Train: Loss=0.0170 Batch_id=117 Accuracy=99.22: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0000, Accuracy: 9943/10000 (99.43%)

Epoch 17
Train: Loss=0.0548 Batch_id=117 Accuracy=99.17: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0000, Accuracy: 9946/10000 (99.46%)

Epoch 18
Train: Loss=0.0413 Batch_id=117 Accuracy=99.23: 100%|██████████| 118/118 [00:55<00:00,  2.13it/s]
Test set: Average loss: 0.0000, Accuracy: 9948/10000 (99.48%)

Epoch 19
Train: Loss=0.0258 Batch_id=117 Accuracy=99.26: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0000, Accuracy: 9948/10000 (99.48%)

Epoch 20
Train: Loss=0.0455 Batch_id=117 Accuracy=99.23: 100%|██████████| 118/118 [00:55<00:00,  2.12it/s]
Test set: Average loss: 0.0000, Accuracy: 9948/10000 (99.48%)

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
