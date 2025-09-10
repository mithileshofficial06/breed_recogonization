import os
from pathlib import Path

class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Upload configuration
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Model configuration
    MODEL_PATH = BASE_DIR / 'models' / 'breed_recognition_model.h5'
    MODEL_INPUT_SIZE = (224, 224)
    
    # Data paths
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    
    # Breed configuration
    BREEDS = {
        'cattle': ['Alamabadi', 'Bargur', 'Gir', 'Kangayam', 'Pulikulam', 'Umblachery'],
        'buffalo': ['Murrah', 'Toda']
    }
    
    ALL_BREEDS = BREEDS['cattle'] + BREEDS['buffalo']
    
    # Prediction configuration
    CONFIDENCE_THRESHOLD = 0.95  # For single prediction
    TOP_K_PREDICTIONS = 3
    
    # Language configuration
    SUPPORTED_LANGUAGES = ['en', 'hi', 'ta']  # English, Hindi, Tamil
    DEFAULT_LANGUAGE = 'en'
    
    # Translation files
    TRANSLATIONS_DIR = BASE_DIR / 'translations'
    
    # Model training configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Data augmentation parameters
    AUGMENTATION_PARAMS = {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS