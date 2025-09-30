Car Brand Classification with PyTorch Lightning
📌 Project Overview

This project implements a car brand classifier using PyTorch Lightning on the Car Brand Dataset
. The aim was to build an efficient classifier with modern deep learning practices while handling dataset challenges.

Dataset Size: 1492 images (~16 MB)

Classes: 16 car brands

Split: Train (1193), Validation (149), Test (150)

⚙️ Approach
Models Tried

Custom CNN: Baseline network.

ResNet-18: Transfer learning with fine-tuning.

EfficientNet_B0: Pre-trained, fine-tuned final classification layer.

Data Augmentation

To mitigate overfitting due to limited data:

Resizing

Horizontal/vertical flips

Random rotations

Color jitter

Affine transformations

Training Setup

Framework: PyTorch Lightning

Loss Function: Cross-Entropy Loss

Optimizer: Adam

Learning Rate Scheduler: StepLR

Batch Size: 32

Epochs: Up to 50

Logging: TensorBoard, Model Checkpoints

📊 Results
Model	Accuracy	Loss
Custom CNN	~15%	High
ResNet-18	~20%	~6+
EfficientNet_B0	~24.7%	~5.03
❌ Why Performance Was Low

Small dataset (1492 images): Too few samples across 16 classes.

Class imbalance: Some brands underrepresented.

Overfitting: Models memorized training data despite augmentations.

Limited diversity: Images not varied enough for robust generalization.

✅ Key Learnings

Pre-trained models (EfficientNet, ResNet) improve results, but data quantity matters most.

PyTorch Lightning made training modular, clean, and experiment-friendly (logging, checkpoints, schedulers).

Augmentation is essential, but cannot fully overcome dataset scarcity.

🚀 Future Improvements

Collect or combine larger car datasets.

Use regularization (Dropout, Mixup, CutMix).

Explore semi-supervised or self-supervised training.

Apply class balancing techniques (oversampling, weighted loss).
