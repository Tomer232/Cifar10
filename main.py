import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import datetime
from PIL import Image
import json

from tensorflow.python.keras.saving.saved_model.load import training_lib

MODEL_CONFIGS = {
    'model_a': {
        'dense_units': 128,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32
    },
    'model_b': {
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': 0.0005,
        'batch_size': 64
    }
}


def create_database():
    database_connection = sqlite3.connect('experiments/cifar10_experiments.db')
    database_cursor = database_connection.cursor()

    database_cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, 
            model_name TEXT,
            hyperparameters TEXT,
            final_accuracy REAL, 
            final_loss REAL,
            training_time REAL,
            model_path TEXT
        )    
    ''')

    database_connection.commit()
    database_connection.close()


def load_and_preprocess_cifar10():
    (training_images, training_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    training_images = training_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    training_labels = keras.utils.to_categorical(training_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    print(f"Training images shape: {training_images.shape}")
    print(f"Test images shape: {test_images.shape}")

    return (training_images, training_labels), (test_images, test_labels), class_names


if __name__ == "__main__":
    print("Creating database...")
    create_database()
    print("Database created successfully")

    print("\nLoading CIFAR-10 data...")
    (train_data, train_labels), (test_data, test_labels), class_names = load_and_preprocess_cifar10()
    print("Data loaded successfully")

    print(f"\nAvailable model configurations : {list(MODEL_CONFIGS.keys())}")
    print(f"Class names: {class_names}")
