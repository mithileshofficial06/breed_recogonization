import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from config import Config

class ImagePreprocessor:
    def __init__(self):
        self.input_size = Config.MODEL_INPUT_SIZE
    
    def preprocess_for_prediction(self, image_path):
        """Preprocess image for model prediction"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Could not read image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.input_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def enhance_image_quality(self, image_path, output_path=None):
        """Enhance image quality for better prediction"""
        try:
            # Open image with PIL
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
            
            if output_path:
                img.save(output_path)
            
            return img
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return None
    
    def preprocess_for_training(self, image_path, label=None):
        """Preprocess image for model training"""
        try:
            # Read and preprocess image
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size)
            img = img.astype(np.float32) / 255.0
            
            if label is not None:
                return img, label
            return img
            
        except Exception as e:
            print(f"Error preprocessing training image: {e}")
            return None
    
    def augment_image(self, image):
        """Apply data augmentation to image"""
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Rotation variations
        for angle in [-15, -10, 10, 15]:
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(image, M, (cols, rows))
            augmented_images.append(rotated)
        
        # Brightness variations
        for beta in [-20, -10, 10, 20]:
            bright = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
            augmented_images.append(bright)
        
        # Zoom variations
        for zoom in [0.9, 1.1]:
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            resized = cv2.resize(image, (new_w, new_h))
            
            if zoom < 1:  # Zoom out - pad image
                pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                zoomed = cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, 
                                          pad_w, w-new_w-pad_w, 
                                          cv2.BORDER_REFLECT)
            else:  # Zoom in - crop image
                crop_h, crop_w = (new_h - h) // 2, (new_w - w) // 2
                zoomed = resized[crop_h:crop_h+h, crop_w:crop_w+w]
            
            zoomed = cv2.resize(zoomed, self.input_size)
            augmented_images.append(zoomed)
        
        return augmented_images
    
    def create_data_generator(self, data_dir, batch_size=32, validation_split=0.2):
        """Create data generators for training"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def validate_image(self, image_path):
        """Validate if image is suitable for processing"""
        try:
            img = Image.open(image_path)
            
            # Check image format
            if img.format.lower() not in ['jpeg', 'jpg', 'png', 'gif']:
                return False, "Unsupported image format"
            
            # Check image size
            if img.size[0] < 100 or img.size[1] < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            # Check if image can be converted to RGB
            if img.mode not in ['RGB', 'L', 'RGBA']:
                return False, "Unsupported color mode"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

# Create global preprocessor instance
image_preprocessor = ImagePreprocessor()