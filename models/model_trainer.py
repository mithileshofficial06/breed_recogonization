import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

from config import Config
from utils.image_preprocessing import ImagePreprocessor

class BreedRecognitionTrainer:
    def __init__(self):
        self.config = Config()
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.history = None
        self.class_names = None
        
    def create_model(self, num_classes):
        """Create the breed recognition model using EfficientNetB0"""
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.MODEL_INPUT_SIZE, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data_dir):
        """Prepare training and validation data generators"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.MODEL_INPUT_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.MODEL_INPUT_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        
        return train_generator, validation_generator
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=str(self.config.MODEL_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                filename=str(self.config.BASE_DIR / 'training_log.csv'),
                append=True
            )
        ]
        
        return callbacks_list
    
    def train_model(self, data_dir, epochs=None, fine_tune=True):
        """Train the breed recognition model"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print("Preparing data...")
        train_gen, val_gen = self.prepare_data(data_dir)
        
        num_classes = len(self.class_names)
        print(f"Found {num_classes} breeds: {self.class_names}")
        
        print("Creating model...")
        self.create_model(num_classes)
        
        print("Model architecture:")
        self.model.summary()
        
        print("Starting initial training...")
        callbacks_list = self.create_callbacks()
        
        # Initial training with frozen base model
        initial_epochs = min(20, epochs // 2)
        history1 = self.model.fit(
            train_gen,
            epochs=initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        if fine_tune:
            print("Fine-tuning with unfrozen layers...")
            # Unfreeze the base model for fine-tuning
            self.model.layers[0].trainable = True
            
            # Use lower learning rate for fine-tuning
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE/10),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
            )
            
            # Continue training
            history2 = self.model.fit(
                train_gen,
                initial_epoch=initial_epochs,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Combine histories
            self.history = self.combine_histories(history1, history2)
        else:
            self.history = history1
        
        print("Training completed!")
        self.save_class_names()
        return self.history
    
    def combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def save_class_names(self):
        """Save class names for later use"""
        class_names_file = self.config.BASE_DIR / 'models' / 'class_names.json'
        with open(class_names_file, 'w') as f:
            json.dump(self.class_names, f)
    
    def load_class_names(self):
        """Load class names"""
        class_names_file = self.config.BASE_DIR / 'models' / 'class_names.json'
        if class_names_file.exists():
            with open(class_names_file, 'r') as f:
                self.class_names = json.load(f)
    
    def evaluate_model(self, data_dir):
        """Evaluate the trained model"""
        if self.model is None:
            print("No model to evaluate. Train a model first.")
            return
        
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.MODEL_INPUT_SIZE,
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        print("Evaluating model...")
        loss, accuracy, top_k_acc = self.model.evaluate(test_generator, verbose=1)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Top-3 Accuracy: {top_k_acc:.4f}")
        
        # Generate predictions for confusion matrix
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names
        ))
        
        # Confusion matrix
        self.plot_confusion_matrix(true_classes, predicted_classes)
        
        return loss, accuracy, top_k_acc
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-K Accuracy
        axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], label='Training')
        axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], label='Validation')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.config.BASE_DIR / 'training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self, true_classes, predicted_classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.config.BASE_DIR / 'confusion_matrix.png')
        plt.show()

def main():
    """Main training function"""
    trainer = BreedRecognitionTrainer()
    
    # Check if data directory exists
    data_dir = Config.RAW_DATA_DIR
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please organize your data in the following structure:")
        print("data/raw/cattle/[breed_name]/[images]")
        print("data/raw/buffalo/[breed_name]/[images]")
        return
    
    # Train model
    print("Starting breed recognition model training...")
    history = trainer.train_model(data_dir, epochs=50, fine_tune=True)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    trainer.evaluate_model(data_dir)
    
    print("Training completed successfully!")
    print(f"Model saved to: {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()