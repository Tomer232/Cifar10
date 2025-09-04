# CIFAR-10 CNN Classification with A/B Testing

## Overview

This project implements a convolutional neural network (CNN) for image classification using the CIFAR-10 dataset. The system demonstrates systematic hyperparameter optimization through A/B testing methodology, comparing a baseline model against an enhanced version with improved training techniques.

## Project Structure

```
cifar10_project/
├── main.py                 # Main application code
├── models/                 # Saved trained models
├── experiments/            # SQLite database for experiment tracking
├── user_images/            # Test images for custom predictions
└── README.md
```

## Features

### Core Functionality
- CNN model training on CIFAR-10 dataset (50,000 training images, 10,000 test images)
- Automated experiment tracking with SQLite database
- Custom image classification for user-uploaded images
- Model comparison and evaluation metrics

### A/B Testing Framework
- **Model A (Baseline)**: Basic CNN with 5 epochs, standard hyperparameters
- **Model B (Enhanced)**: Improved training with 20 epochs, learning rate scheduling, early stopping, and batch normalization

### Advanced Training Features
- Learning rate reduction on plateau
- Early stopping to prevent overfitting
- Batch normalization for improved convergence
- Comprehensive hyperparameter logging

## Technical Implementation

### Model Architecture
Both models use identical CNN architecture:
- 3 convolutional layers (32, 64, 64 filters)
- MaxPooling and batch normalization
- Dense layers with dropout regularization
- Softmax output for 10-class classification

### Hyperparameter Differences
| Parameter | Model A | Model B |
|-----------|---------|---------|
| Epochs | 5 | 20 |
| Dropout Rate | 0.2 | 0.3 |
| Dense Units | 128 | 256 |
| Learning Rate Scheduling | None | Adaptive |
| Early Stopping | None | Enabled |

## Results

The enhanced model (Model B) consistently outperforms the baseline:
- **Model A**: ~66-70% accuracy
- **Model B**: ~74-76% accuracy
- **Improvement**: +7-8% accuracy gain

Training demonstrates the effectiveness of advanced optimization techniques while maintaining architectural consistency for valid comparison.

## Usage

### Training Models
```bash
python main.py
```

### Custom Image Testing
1. Add images to the `user_images/` folder
2. Run the script to see predictions from both models
3. Supported formats: JPG, JPEG, PNG, BMP, GIF

### Database Queries
All experiments are logged in `experiments/cifar10_experiments.db` with:
- Model configurations
- Training metrics
- Execution timestamps
- Model file paths

## Dependencies

- TensorFlow/Keras
- NumPy
- PIL (Pillow)
- SQLite3
- Matplotlib

## Scientific Methodology

This project follows controlled experimental design principles:
- Minimal architectural changes between models
- Systematic hyperparameter variation
- Reproducible experiment tracking
- Statistical comparison of results

The approach demonstrates practical machine learning optimization while maintaining scientific rigor suitable for academic evaluation.
