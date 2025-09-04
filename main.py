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

MODEL_CONFIGS = {
    'model_a': {
        'dense_units': 128,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 5
    },
    'model_b': {
        'dense_units': 256,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20
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


def create_cnn_model(model_config):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(model_config['dense_units'], activation='relu'),
        layers.Dropout(model_config['dropout_rate']),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=model_config['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_enhanced_model(model_name, model_config, training_data, training_labels, test_data, test_labels):
    print(f"\n --- training {model_name} (enhanced) ---")
    model = create_cnn_model(model_config)

    start_time = datetime.datetime.now()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    training_history = model.fit(
        training_data, training_labels, batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        validation_data=(test_data, test_labels),
        callbacks=callbacks,
        verbose=1
    )

    end_time = datetime.datetime.now()
    training_duration = abs((start_time - end_time).total_seconds())

    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"{model_name} - final accuracy: {test_accuracy:.4f}")

    return model, test_accuracy, test_loss, training_duration, training_history

def train_and_evaluate_model(model_name, model_config, training_data, training_labels, test_data, test_labels):
    print(f"\n--- Training {model_name} ---")
    model = create_cnn_model(model_config)

    start_time = datetime.datetime.now()

    training_history = model.fit(
        training_data,
        training_labels,
        batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        validation_data=(test_data, test_labels),
        verbose=1
    )

    end_time = datetime.datetime.now()
    training_duration = abs((start_time - end_time). total_seconds())

    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)

    print(f"{model_name} - Final accuracy: {test_accuracy:.4f}")

    return model, test_accuracy, test_loss, training_duration, training_history


def save_experiment_to_database(model_name, model_config, final_accuracy, final_loss, training_time):
    database_connection = sqlite3.connect('experiments/cifar10_experiments.db')
    database_cursor = database_connection.cursor()

    hyperparameters_json = json.dumps(model_config)
    current_timestamp = datetime.datetime.now().isoformat()
    model_save_path = f"models/{model_name}_{current_timestamp.replace(':', '-')}.h5"

    database_cursor.execute('''
        INSERT INTO experiments
        (timestamp, model_name, hyperparameters, final_accuracy, final_loss, training_time, model_path)
        VALUES (?, ?, ?, ?, ?, ?, ?) 
    ''', (current_timestamp, model_name, hyperparameters_json, final_accuracy, final_loss, training_time, model_save_path))

    database_connection.commit()
    database_connection.close()

    print(f"Experiments {model_name} saved to DB")
    return model_save_path


def view_experiment_results():
    database_connection = sqlite3.connect('experiments/cifar10_experiments.db')
    database_cursor = database_connection.cursor()

    database_cursor.execute('SELECT * FROM experiments ORDER BY final_accuracy DESC')
    results = database_cursor.fetchall()

    print(f"=== EXPERIMENT RESULTS (Best to worst) ===")
    for row in results:
        experiments_id, timestamp, model_name, hyperparams, accuracy, loss, duration, model_path = row
        print(f"Model: {model_name} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f} | Duration: {duration:.1f}")
        hyperparams_dict = json.loads(hyperparams)
        print(f" Config: {hyperparams_dict}")
        print("-" * 60)

    database_connection.close()


def preprocess_user_image(image_path):
    try:
        user_images = Image.open(image_path)

        if user_images.mode != 'RGB':
            user_images = user_images.convert('RGB')

        user_images = user_images.resize((32, 32))

        image_array = np.array(user_images)
        image_array = image_array.astype('float32') / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as error:
        print(f"Error processing image {image_path}: {error}")
        return None


def predict_user_image(model, class_names):
    user_images_folder = 'user_images'

    if not os.path.exists(user_images_folder):
        print(f"Creating {user_images_folder} folder...")
        os.makedirs(user_images_folder)
        print(f"Please add your images to the {user_images_folder} folder and run again")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []

    for filename in os.listdir(user_images_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)

    if not image_files:
        print(f"No images found in {user_images_folder} folder")
        print(f"supported formats: {image_extensions}")
        return

    print(f"\n=== Predicting {len(image_files)} user images ===")

    for image_file in image_files:
        image_path = os.path.join(user_images_folder, image_file)
        processed_image = preprocess_user_image(image_path)

        if processed_image is not None:
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]
            predicted_class = class_names[predicted_class_index]

            print(f"image: {image_file}")
            print(f"Prediction: {predicted_class} (confidence: {confidence:.2f})")
            print(f" Top 3 predictions: ")

            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            for i, idx in enumerate(top_3_indices):
                print(f"     {i+1}. {class_names[idx]}: {predictions[0][idx]:.2f}")
            print("-" * 35)


def compare_models_on_user_images(model_a, model_b, class_names):
    user_images_folder = 'user_images'

    if not os.path.exists(user_images_folder):
        print(f"no {user_images_folder} folder found")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []

    for filename in os.listdir(user_images_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)

    if not image_files:
        print(f"No images found in {user_images_folder} folder.")
        return

    print(f"\n=== Model comparison on {len(image_files)} user images ===")

    for image_file in image_files:
        image_path = os.path.join(user_images_folder, image_file)
        processed_image = preprocess_user_image(image_path)

        if processed_image is not None:
            # Get predictions from both models
            pred_a = model_a.predict(processed_image, verbose=0)
            pred_b = model_b.predict(processed_image, verbose=0)

            class_a = np.argmax(pred_a[0])
            class_b = np.argmax(pred_b[0])
            confidence_a = pred_a[0][class_a]
            confidence_b = pred_b[0][class_b]

            print(f"\nImage: {image_file}")
            print(f"  Model A (Basic): {class_names[class_a]} (confidence: {confidence_a:.2f})")
            print(f"  Model B (Enhanced): {class_names[class_b]} (confidence: {confidence_b:.2f})")


if __name__ == "__main__":
    print("Creating database...")
    create_database()
    print("Database created successfully")

    print("\nLoading CIFAR-10 data...")
    (training_data, training_labels), (test_data, test_labels), class_names = load_and_preprocess_cifar10()
    print("Data loaded successfully")

    best_model = None
    best_accuracy = 0

    print(f"\n=== Training model A (basic) ===")
    model_a, accuracy_a, loss_a, duration_a, _ = train_and_evaluate_model('model_a', MODEL_CONFIGS['model_a'], training_data, training_labels, test_data, test_labels)
    model_path_a = save_experiment_to_database('model_a', MODEL_CONFIGS['model_a'], accuracy_a, loss_a, duration_a)

    if not os.path.exists('models'):
        os.makedirs('models')
    model_a.save(model_path_a)
    print(f"model a completed in {duration_a:.2f} seconds with {accuracy_a:.4f} accuracy")

    if accuracy_a > best_accuracy:
        best_accuracy = accuracy_a
        best_model = model_a

    print(f"\n=== Training model B (basic) ===")
    model_b, accuracy_b, loss_b, duration_b, _ = train_enhanced_model('model_b', MODEL_CONFIGS['model_b'], training_data, training_labels, test_data, test_labels)
    model_path_b = save_experiment_to_database('model_b', MODEL_CONFIGS['model_b'], accuracy_b, loss_b, duration_b)
    print(f"model b completed in {duration_b:.2f} seconds with {accuracy_b:.4f} accuracy")

    if accuracy_b > best_accuracy:
        best_accuracy = accuracy_b
        best_model = model_b

    print(f"\n=== MODEL COMPARISON ===")
    print(f"Model A (Basic): {accuracy_a:.4f} accuracy in {duration_a:.1f}s")
    print(f"Model B (Enhanced): {accuracy_b:.4f} accuracy in {duration_b:.1f}s")
    print(f"Improvement: +{((accuracy_b - accuracy_a) * 100):.2f}% accuracy")

    view_experiment_results()

    print(f"\n=== Comparing both models on user images ===")
    compare_models_on_user_images(model_a, model_b, class_names)

