import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import cv2
from PIL import Image

from config import Config
from utils.image_preprocessing import ImagePreprocessor

class BreedPredictor:
    def __init__(self):
        self.config = Config()
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.class_names = None
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if self.config.MODEL_PATH.exists():
                self.model = tf.keras.models.load_model(str(self.config.MODEL_PATH))
                print(f"Model loaded successfully from {self.config.MODEL_PATH}")
            else:
                print(f"Model file not found: {self.config.MODEL_PATH}")
                print("Please train the model first using model_trainer.py")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def load_class_names(self):
        """Load class names from file"""
        class_names_file = self.config.BASE_DIR / 'models' / 'class_names.json'
        try:
            if class_names_file.exists():
                with open(class_names_file, 'r') as f:
                    self.class_names = json.load(f)
                print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
            else:
                # Fallback to config breeds if file doesn't exist
                self.class_names = self.config.ALL_BREEDS
                print("Using fallback class names from config")
        except Exception as e:
            print(f"Error loading class names: {e}")
            self.class_names = self.config.ALL_BREEDS
    
    def predict_breed(self, image_path):
        """
        Predict breed from image
        Returns: dict with prediction results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train the model first.',
                'predictions': []
            }
        
        try:
            # Preprocess image
            preprocessed_image = self.preprocessor.preprocess_for_prediction(image_path)
            if preprocessed_image is None:
                return {
                    'success': False,
                    'error': 'Failed to preprocess image',
                    'predictions': []
                }
            
            print(f"Preprocessed image shape: {preprocessed_image.shape}")
            
            # Make prediction
            predictions = self.model.predict(preprocessed_image, verbose=0)
            print(f"Raw model predictions shape: {predictions.shape}")
            print(f"Raw predictions: {predictions}")
            
            # Handle predictions array properly
            if len(predictions.shape) > 1:
                confidence_scores = predictions[0]  # Get first batch item
            else:
                confidence_scores = predictions
            
            print(f"Confidence scores shape: {confidence_scores.shape}")
            print(f"Confidence scores: {confidence_scores}")
            print(f"Number of class names: {len(self.class_names)}")
            
            # Ensure we have enough class names for all predictions
            if len(confidence_scores) > len(self.class_names):
                print(f"WARNING: Model predicts {len(confidence_scores)} classes but only {len(self.class_names)} class names available")
                # Pad class names if needed
                for i in range(len(self.class_names), len(confidence_scores)):
                    self.class_names.append(f"Unknown_Class_{i}")
            
            # Get top predictions
            top_indices = np.argsort(confidence_scores)[::-1]
            print(f"Top indices: {top_indices}")
            
            results = []
            num_predictions = min(self.config.TOP_K_PREDICTIONS, len(top_indices), len(self.class_names))
            
            for i in range(num_predictions):
                idx = top_indices[i]
                
                # Safety check for index bounds
                if idx < len(self.class_names):
                    breed_name = self.class_names[idx]
                else:
                    breed_name = f"Class_{idx}"
                    print(f"WARNING: Index {idx} is out of bounds for class_names. Using fallback name.")
                
                # Safety check for confidence scores bounds
                if idx < len(confidence_scores):
                    confidence = float(confidence_scores[idx])
                else:
                    confidence = 0.0
                    print(f"WARNING: Index {idx} is out of bounds for confidence_scores.")
                
                results.append({
                    'breed': breed_name,
                    'confidence': confidence,
                    'confidence_percent': round(confidence * 100, 2)
                })
                
                print(f"Prediction {i+1}: {breed_name} - {confidence:.4f} ({confidence*100:.2f}%)")
            
            if not results:
                return {
                    'success': False,
                    'error': 'No valid predictions could be generated',
                    'predictions': []
                }
            
            # Check if top prediction is highly confident
            is_single_prediction = (
                len(results) > 0 and 
                results[0]['confidence'] >= self.config.CONFIDENCE_THRESHOLD
            )
            
            return {
                'success': True,
                'is_single_prediction': is_single_prediction,
                'top_prediction': results[0] if results else None,
                'predictions': results,
                'image_processed': True
            }
            
        except Exception as e:
            import traceback
            print(f"Exception in predict_breed: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'predictions': []
            }
    
    def predict_with_preprocessing_variants(self, image_path):
        """
        Try prediction with different preprocessing approaches
        """
        results = []
        
        # Original prediction
        result = self.predict_breed(image_path)
        if result['success']:
            results.append(('original', result))
        
        # Try with enhanced image
        try:
            enhanced_image = self.preprocessor.enhance_image_quality(image_path)
            if enhanced_image:
                # Save temporarily
                temp_path = self.config.UPLOAD_FOLDER / 'temp_enhanced.jpg'
                enhanced_image.save(temp_path)
                
                enhanced_result = self.predict_breed(temp_path)
                if enhanced_result['success']:
                    results.append(('enhanced', enhanced_result))
                
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
        
        except Exception as e:
            print(f"Enhancement failed: {e}")
        
        # Return best result (highest confidence)
        if not results:
            return result
        
        best_result = max(results, key=lambda x: x[1]['predictions'][0]['confidence'] if x[1]['predictions'] else 0)
        return best_result[1]
    
    def get_prediction_explanation(self, predictions, language='en'):
        """
        Generate explanation text for predictions
        """
        if not predictions:
            return "No predictions available."
        
        explanations = {
            'en': {
                'single_high': "The model is highly confident (>95%) that this is a {breed}.",
                'multiple': "The model suggests the following possible breeds:",
                'top_match': "Most likely: {breed} ({confidence}% confidence)",
                'alternatives': "Other possibilities:"
            },
            'hi': {
                'single_high': "मॉडल को अत्यधिक विश्वास (>95%) है कि यह {breed} है।",
                'multiple': "मॉडल निम्नलिखित संभावित नस्लों का सुझाव देता है:",
                'top_match': "सबसे संभावित: {breed} ({confidence}% विश्वसनीयता)",
                'alternatives': "अन्य संभावनाएं:"
            },
            'ta': {
                'single_high': "மாதிரி அதிக நம்பிக்கையுடன் (>95%) இது {breed} என்று கூறுகிறது।",
                'multiple': "மாதிரி பின்வரும் சாத்தியமான இனங்களை பரிந்துரைக்கிறது:",
                'top_match': "மிகவும் சாத்தியமான: {breed} ({confidence}% நம்பகத்தன்மை)",
                'alternatives': "மற்ற சாத்தியங்கள்:"
            }
        }
        
        lang_texts = explanations.get(language, explanations['en'])
        
        if len(predictions) == 1 and predictions[0]['confidence'] > 0.95:
            return lang_texts['single_high'].format(
                breed=predictions[0]['breed']
            )
        else:
            explanation = lang_texts['multiple'] + "\n"
            explanation += lang_texts['top_match'].format(
                breed=predictions[0]['breed'],
                confidence=predictions[0]['confidence_percent']
            )
            
            if len(predictions) > 1:
                explanation += "\n" + lang_texts['alternatives']
                for pred in predictions[1:]:
                    explanation += f"\n- {pred['breed']} ({pred['confidence_percent']}%)"
            
            return explanation
    
    def batch_predict(self, image_paths):
        """
        Predict breeds for multiple images
        """
        results = []
        for image_path in image_paths:
            result = self.predict_breed(image_path)
            results.append({
                'image_path': str(image_path),
                'result': result
            })
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if self.model is None:
            return {
                'loaded': False,
                'error': 'Model not loaded'
            }
        
        try:
            return {
                'loaded': True,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'num_classes': len(self.class_names) if self.class_names else 0,
                'class_names': self.class_names,
                'total_params': self.model.count_params(),
                'model_path': str(self.config.MODEL_PATH)
            }
        except Exception as e:
            return {
                'loaded': False,
                'error': f'Error getting model info: {str(e)}'
            }
    
    def validate_prediction_confidence(self, predictions):
        """
        Validate and categorize prediction confidence
        """
        if not predictions:
            return 'no_prediction'
        
        top_confidence = predictions[0]['confidence']
        
        if top_confidence >= 0.95:
            return 'very_high'
        elif top_confidence >= 0.80:
            return 'high'
        elif top_confidence >= 0.60:
            return 'medium'
        elif top_confidence >= 0.40:
            return 'low'
        else:
            return 'very_low'
    
    def get_confidence_message(self, confidence_level, language='en'):
        """
        Get user-friendly confidence message
        """
        messages = {
            'en': {
                'very_high': 'Very High Confidence - The model is very certain about this prediction.',
                'high': 'High Confidence - The model is quite confident about this prediction.',
                'medium': 'Medium Confidence - The model has reasonable confidence in this prediction.',
                'low': 'Low Confidence - The model is uncertain. Consider multiple possibilities.',
                'very_low': 'Very Low Confidence - The model is very uncertain about this prediction.',
                'no_prediction': 'No prediction could be made.'
            },
            'hi': {
                'very_high': 'बहुत उच्च विश्वसनीयता - मॉडल इस भविष्यवाणी के बारे में बहुत निश्चित है।',
                'high': 'उच्च विश्वसनीयता - मॉडल इस भविष्यवाणी के बारे में काफी आश्वस्त है।',
                'medium': 'मध्यम विश्वसनीयता - मॉडल में इस भविष्यवाणी में उचित विश्वास है।',
                'low': 'कम विश्वसनीयता - मॉडल अनिश्चित है। कई संभावनाओं पर विचार करें।',
                'very_low': 'बहुत कम विश्वसनीयता - मॉडल इस भविष्यवाणी के बारे में बहुत अनिश्चित है।',
                'no_prediction': 'कोई भविष्यवाणी नहीं की जा सकी।'
            },
            'ta': {
                'very_high': 'மிக உயர்ந்த நம்பகத்தன்மை - மாதிரி இந்த கணிப்பு குறித்து மிகவும் உறுதியாக உள்ளது.',
                'high': 'உயர் நம்பகத்தன்மை - மாதிரி இந்த கணிப்பு குறித்து மிகவும் நம்பிக்கையுடன் உள்ளது.',
                'medium': 'நடுத்தர நம்பகத்தன்மை - மாதிரிக்கு இந்த கணிப்பில் நியாயமான நம்பிக்கை உள்ளது.',
                'low': 'குறைந்த நம்பகத்தன்மை - மாதிரி நிச்சயமற்றது. பல சாத்தியங்களைக் கருத்தில் கொள்ளுங்கள்.',
                'very_low': 'மிக குறைந்த நம்பகத்தன்மை - மாதிரி இந்த கணிப்பு குறித்து மிகவும் நிச்சயமற்றது.',
                'no_prediction': 'எந்த கணிப்பும் செய்ய முடியவில்லை.'
            }
        }
        
        return messages.get(language, messages['en']).get(confidence_level, messages['en']['no_prediction'])

# Create global predictor instance
breed_predictor = BreedPredictor()

def test_predictor(image_path):
    """Test function for the predictor"""
    predictor = BreedPredictor()
    
    print(f"Testing prediction for: {image_path}")
    print("Model Info:", predictor.get_model_info())
    
    result = predictor.predict_breed(image_path)
    print("Prediction Result:", result)
    
    if result['success'] and result['predictions']:
        confidence_level = predictor.validate_prediction_confidence(result['predictions'])
        print(f"Confidence Level: {confidence_level}")
        
        for lang in ['en', 'hi', 'ta']:
            explanation = predictor.get_prediction_explanation(result['predictions'], lang)
            confidence_msg = predictor.get_confidence_message(confidence_level, lang)
            print(f"\n{lang.upper()}:")
            print(f"Explanation: {explanation}")
            print(f"Confidence: {confidence_msg}")

if __name__ == "__main__":
    # Test the predictor if run directly
    import sys
    if len(sys.argv) > 1:
        test_predictor(sys.argv[1])
    else:
        print("Usage: python model_predictor.py <image_path>")