# ResNet50 on CIFAR-10 Classification

This project implements image classification on the CIFAR-10 dataset using the ResNet50 model from TensorFlow/Keras. The model is trained in two phases: first, training only the custom classification head, and then fine-tuning the entire model after unfreezing the ResNet50 base.

## Dataset
CIFAR-10 consists of 60,000 32x32 color images in 10 different classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

For this project, we use a subset of 10,000 images for training and 10,000 images for testing.

## Steps in the Project
1. **Load and Preprocess Data:**
   - Normalize images to a range of [0,1].
   - One-hot encode labels.
   
2. **Build Model Using ResNet50:**
   - Use ResNet50 (pre-trained on ImageNet) as the base model.
   - Freeze base model layers initially.
   - Add a custom head with GlobalAveragePooling2D, Dense layers, and Dropout.
   
3. **Train the Model:**
   - First, train only the custom classification head for 10 epochs.
   - Unfreeze the base model and fine-tune the entire model for another 10 epochs.

4. **Evaluate Performance:**
   - Achieved **training accuracy of 89%**.
   - Achieved **test accuracy of 73.56%**.
   
5. **Visualization:**
   - Display sample images from CIFAR-10.
   - Plot training accuracy and loss over epochs.

## Installation & Requirements
Ensure you have the following installed:
```bash
pip install tensorflow numpy matplotlib
```

## Running the Code
To execute the model training and evaluation, run the script:
```bash
python resnet50_cifar10.py
```

## Results
| Metric         | Value  |
|---------------|--------|
| Train Accuracy | 89%   |
| Test Accuracy  | 73.56% |

## Future Improvements
- Train for more epochs to further improve test accuracy.
- Experiment with different architectures and hyperparameters.
- Apply data augmentation techniques to enhance generalization.



