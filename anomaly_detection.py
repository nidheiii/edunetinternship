import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MIN_IMAGES_PER_CLASS = 20


def create_model():
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def prepare_data(train_dir, validation_dir):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load and augment training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, validation_generator


def train_model(model, train_generator, validation_generator):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    return model, history


def check_anomaly(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Anomaly detected"
    else:
        return "No anomaly detected"


def create_sample_image(path, is_anomaly=False):
    plt.figure(figsize=(3, 3))
    if is_anomaly:
        plt.imshow(np.random.rand(100, 100, 3))
    else:
        plt.imshow(np.ones((100, 100, 3)) * 0.5)
    plt.axis('off')
    plt.savefig(path)
    plt.close()


def create_directory_structure(base_dir):
    for split in ['train', 'validation']:
        for category in ['normal', 'anomaly']:
            dir_path = base_dir / 'data' / split / category
            dir_path.mkdir(parents=True, exist_ok=True)

            # Create sample images for each category
            for i in range(MIN_IMAGES_PER_CLASS):
                img_path = dir_path / f'sample_{i}.png'
                create_sample_image(img_path, is_anomaly=(category == 'anomaly'))

    print(f"Sample directory structure created with {MIN_IMAGES_PER_CLASS} images per class.")


def check_data_dirs(train_dir, validation_dir):
    for dir_path in [train_dir, validation_dir]:
        for category in ['normal', 'anomaly']:
            category_path = dir_path / category
            if not category_path.exists():
                raise FileNotFoundError(f"Directory not found: {category_path}")

            image_files = list(category_path.glob('*.png')) + list(category_path.glob('*.jpg')) + list(
                category_path.glob('*.jpeg'))
            if len(image_files) < MIN_IMAGES_PER_CLASS:
                raise ValueError(
                    f"Not enough images in {category_path}. Found {len(image_files)}, need at least {MIN_IMAGES_PER_CLASS}.")


# Main execution
if __name__ == "__main__":
    # Use Path for cross-platform compatibility
    base_dir = Path(r"C:\Users\ASUSH\PycharmProjects\CNN Project 1 - Copy")
    train_dir = base_dir / "data" / "train"
    validation_dir = base_dir / "data" / "validation"

    # Create directory structure and sample images if they don't exist
    if not train_dir.exists() or not validation_dir.exists():
        print("Creating sample directory structure and images...")
        create_directory_structure(base_dir)

    try:
        # Check if the data directories have enough images
        check_data_dirs(train_dir, validation_dir)

        # Create and train the model
        model = create_model()
        train_generator, validation_generator = prepare_data(str(train_dir), str(validation_dir))
        trained_model, history = train_model(model, train_generator, validation_generator)

        # Save the trained model
        model_save_path = base_dir / "anomaly_detection_model.keras"
        tf.keras.models.save_model(trained_model, str(model_save_path), save_format='keras')
        print(f"Model saved to {model_save_path}")

        # Example usage of anomaly detection
        test_image_path = base_dir / "test_image.jpg"
        if not test_image_path.exists():
            print("Creating a test image...")
            create_sample_image(test_image_path, is_anomaly=True)

        result = check_anomaly(trained_model, str(test_image_path))
        print(result)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that the data directories exist and contain the necessary subdirectories.")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Please ensure that each category has at least {MIN_IMAGES_PER_CLASS} images.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

