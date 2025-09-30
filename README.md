# 🚗 Car Brand Classification with PyTorch Lightning

---

### 📌 Project Overview

This project implements a car brand classifier using **PyTorch Lightning** on the Car Brand Dataset. The aim was to build an efficient classifier with modern deep learning practices while handling dataset challenges.

-   **Dataset Size**: 1492 images (~16 MB)
-   **Classes**: 16 car brands
-   **Split**: Train (1193), Validation (149), Test (150)

---

### ⚙️ Approach

#### Models Tried
-   **Custom CNN**: A baseline network to establish initial performance.
-   **ResNet-18**: Transfer learning with fine-tuning on the pre-trained model.
-   **EfficientNet_B0**: Utilized the pre-trained model and fine-tuned the final classification layer.

#### Data Augmentation
To mitigate overfitting due to the limited dataset size, the following augmentations were applied:
-   Resizing
-   Horizontal/vertical flips
-   Random rotations
-   Color jitter
-   Affine transformations

#### Training Setup
-   **Framework**: PyTorch Lightning
-   **Loss Function**: Cross-Entropy Loss
-   **Optimizer**: Adam
-   **Learning Rate Scheduler**: `StepLR`
-   **Batch Size**: 32
-   **Epochs**: Up to 50
-   **Logging**: TensorBoard and Model Checkpoints

---

### 📊 Results

| Model         | Accuracy | Loss  |
|---------------|----------|-------|
| Custom CNN    | ~15%     | High  |
| ResNet-18     | ~20%     | ~6.0+ |
| EfficientNet_B0 | ~24.7%   | ~5.03 |

---

### ❌ Why Performance Was Low

1.  **Small Dataset**: With only 1492 images, there were too few samples across 16 classes for the models to learn effectively.
2.  **Class Imbalance**: Some car brands were significantly underrepresented, biasing the model.
3.  **Overfitting**: Despite using augmentations, the models tended to memorize the training data rather than generalize.
4.  **Limited Diversity**: The images lacked sufficient variation in angles, lighting, and backgrounds for robust feature extraction.

---

### ✅ Key Learnings

-   Pre-trained models like **EfficientNet** and **ResNet** provide a significant boost over custom architectures, but data quantity remains the most critical factor.
-   **PyTorch Lightning** streamlines the training process, making it modular, clean, and easy to experiment with features like logging, checkpoints, and schedulers.
-   Data augmentation is essential for small datasets but cannot fully compensate for a fundamental lack of data.

---

### 🚀 Future Improvements

-   **Data Expansion**: Collect more data or combine this dataset with larger, more diverse car datasets.
-   **Advanced Regularization**: Implement techniques like Dropout, Mixup, or CutMix to further combat overfitting.
-   **Modern Training Methods**: Explore semi-supervised or self-supervised learning to leverage unlabeled data.
-   **Class Balancing**: Apply techniques such as oversampling underrepresented classes or using a weighted loss function.
