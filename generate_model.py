#!/usr/bin/env python3
"""
Model Generation Script for Breed Recognition System
===================================================

This script generates the breed_recognition_model.h5 file by training
a CNN model on cattle breed data or creating a pre-trained model structure.

Usage:
    python generate_model.py [--train] [--use-pretrained]
    
Options:
    --train: Train the model from scratch using data in data/raw/
    --use-pretrained: Create model using transfer learning from VGG16/ResNet50
    --quick: Generate a basic model structure for testing (default)

Requirements:
    - TensorFlow 2.x
    - Organized image data in data/raw/ folder structure
    - At least 100 images per breed class for training
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        import cv2
        from PIL import Image
        print(f"✓ TensorFlow version: {tf.__version__}")
        print(f"✓ OpenCV and PIL available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install tensorflow opencv-python Pillow numpy")
        return False

def create_basic_model():
    """Create a basic CNN model structure for testing"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print("Creating basic CNN model structure...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(224, 224, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer for 9 breed classes
        layers.Dense(9, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def create_transfer_learning_model():
    """Create a model using transfer learning from pre-trained networks"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import VGG16
    
    print("Creating transfer learning model with VGG16 base...")
    
    # Load pre-trained VGG16 model without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Custom dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'), 
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer for 9 classes
        layers.Dense(9, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def load_and_prepare_data():
    """Load and prepare training data"""
    data_dir = Path('data/raw')
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print("Please organize your data as follows:")
        print("data/raw/")
        print("  ├── Alamabadi/")
        print("  ├── Bargur/")
        print("  ├── Gir/")
        print("  └── ...")
        return None, None, None, None
    
    # Check breed folders
    expected_breeds = [
        'Alamabadi', 'Bargur', 'Gir', 'Kangayam', 
        'Pulikulam', 'Umblachery', 'Buffalo', 'Murrah', 'Toda'
    ]
    
    available_breeds = []
    for breed in expected_breeds:
        breed_path = data_dir / breed
        if breed_path.exists():
            image_count = len(list(breed_path.glob('*.jpg')) + list(breed_path.glob('*.png')))
            if image_count > 0:
                available_breeds.append(breed)
                print(f"✓ Found {image_count} images for {breed}")
            else:
                print(f"⚠ No images found for {breed}")
        else:
            print(f"⚠ Missing breed folder: {breed}")
    
    if len(available_breeds) < 3:
        print("✗ Need at least 3 breeds with images to train")
        return None, None, None, None
    
    # Use TensorFlow's image_dataset_from_directory
    import tensorflow as tf
    
    print(f"Loading dataset from {data_dir}")
    
    # Create training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # Create validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation", 
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")
    
    # Normalize pixel values
    normalization_layer = tf.keras.utils.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names, len(available_breeds)

def train_model_from_data(use_transfer_learning=False):
    """Train model using actual data"""
    print("=" * 60)
    print("TRAINING MODEL FROM DATA")
    print("=" * 60)
    
    # Load data
    train_ds, val_ds, class_names, num_classes = load_and_prepare_data()
    
    if train_ds is None:
        print("✗ Cannot load training data")
        return False
    
    # Create model
    if use_transfer_learning:
        model = create_transfer_learning_model()
    else:
        model = create_basic_model()
    
    print(f"✓ Created model for {num_classes} classes")
    model.summary()
    
    # Callbacks
    import tensorflow as tf
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/breed_recognition_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    print("Starting training...")
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/breed_recognition_model.h5')
    print("✓ Model saved as breed_recognition_model.h5")
    
    # Print training results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    
    return True

def create_demo_model():
    """Create a demo model with random weights for testing"""
    print("=" * 60)
    print("CREATING DEMO MODEL")
    print("=" * 60)
    
    print("Creating demo model with basic architecture...")
    
    model = create_basic_model()
    
    print("✓ Demo model created")
    model.summary()
    
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save the model
    model_path = 'models/breed_recognition_model.h5'
    model.save(model_path)
    
    print(f"✓ Demo model saved as {model_path}")
    print("⚠ Note: This is a demo model with random weights")
    print("   For production use, train with actual data using --train flag")
    
    return True

def validate_model():
    """Validate the generated model"""
    model_path = 'models/breed_recognition_model.h5'
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    try:
        import tensorflow as tf
        
        print("Validating model...")
        model = tf.keras.models.load_model(model_path)
        
        print("✓ Model loaded successfully")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        print(f"  - Parameters: {model.count_params():,}")
        
        # Test prediction with dummy data
        import numpy as np
        dummy_input = np.random.random((1, 224, 224, 3))
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✓ Test prediction successful")
        print(f"  - Output shape: {prediction.shape}")
        print(f"  - Output sum: {prediction.sum():.4f} (should be ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False

def create_class_mapping():
    """Create class index mapping file"""
    breeds = [
        'Alamabadi', 'Bargur', 'Gir', 'Kangayam', 
        'Pulikulam', 'Umblachery', 'Buffalo', 'Murrah', 'Toda'
    ]
    
    class_mapping = {i: breed for i, breed in enumerate(breeds)}
    
    # Save to JSON file
    import json
    
    mapping_path = 'models/class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"✓ Class mapping saved to {mapping_path}")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'data/raw',
        'data/processed', 
        'data/augmented',
        'uploads'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate breed recognition model"
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model from data in data/raw/ folder'
    )
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        help='Use transfer learning with pre-trained weights'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Generate basic demo model for testing (default)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing model'
    )
    
    args = parser.parse_args()
    
    print("Breed Recognition Model Generator")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Validate existing model only
    if args.validate_only:
        if validate_model():
            print("✓ Model validation successful")
            return 0
        else:
            print("✗ Model validation failed")
            return 1
    
    success = False
    
    # Training mode
    if args.train:
        success = train_model_from_data(args.use_pretrained)
    else:
        # Default: create demo model
        success = create_demo_model()
    
    if success:
        # Create class mapping
        create_class_mapping()
        
        # Validate the generated model
        if validate_model():
            print("\n" + "=" * 50)
            print("✓ MODEL GENERATION SUCCESSFUL!")
            print("=" * 50)
            print("Generated files:")
            print("  - models/breed_recognition_model.h5")
            print("  - models/class_mapping.json")
            print("\nYou can now run the Flask application:")
            print("  python app.py")
            return 0
        else:
            print("✗ Model validation failed")
            return 1
    else:
        print("✗ Model generation failed")
        return 1

if __name__ == "__main__":
    exit(main())