#!/usr/bin/env python3
"""
Breed Recognition Flask Application - Using models/model_predictor.py
====================================================================
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))  # Add models folder to path

from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Import your predictor class from models folder
try:
    from models.model_predictor import BreedPredictor
    # Create predictor instance
    breed_predictor = BreedPredictor()
    print("✅ Model predictor imported successfully from models folder")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure model_predictor.py is in the models/ directory")
    sys.exit(1)

# Configuration
class Config:
    SECRET_KEY = '7f7bc40d677d4cd6d3e28834b8b3c625d2f3b59f0c0cf04e8a78a75c3295d40a'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_language():
    """Get current language"""
    return session.get('language', 'en')

def load_translations(language='en'):
    """Load translation file for given language"""
    try:
        translations_path = Path('translations') / f'{language}.json'
        if translations_path.exists():
            with open(translations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Fallback to English if language file not found
            fallback_path = Path('translations') / 'en.json'
            if fallback_path.exists():
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
    except Exception as e:
        print(f"Error loading translations: {e}")
        return {}

def get_text(key):
    """Get translated text for current language"""
    language = get_language()
    translations = load_translations(language)
    return translations.get(key, key)

@app.context_processor
def utility_processor():
    return dict(get_language=get_language, get_text=get_text)

@app.route('/')
def index():
    """Home page"""
    print("🏠 Index route accessed")
    return render_template('index.html')

@app.route('/set_language/<language>')
def set_language(language):
    """Set the current language"""
    supported_languages = ['en', 'hi', 'ta']
    
    if language in supported_languages:
        session['language'] = language
    
    return redirect(request.referrer or url_for('index'))

@app.route('/debug/routes')
def list_routes():
    """Debug endpoint to list all available routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({'routes': routes})

@app.route('/upload', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and breed prediction using models/model_predictor.py"""
    print(f"📝 Upload route accessed with method: {request.method}")
    
    if request.method == 'GET':
        return jsonify({
            'message': 'Upload endpoint is working. Use POST method with file.',
            'supported_methods': ['POST'],
            'expected_form_field': 'file'
        })
    
    try:
        print(f"📋 Request files: {list(request.files.keys())}")
        print(f"📋 Request form: {dict(request.form)}")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            print("❌ No 'file' key in request.files")
            return jsonify({'error': 'No file uploaded', 'received_keys': list(request.files.keys())}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"📁 Received file: {file.filename}")
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Check if model is loaded in predictor
        try:
            model_info = breed_predictor.get_model_info()
            if not model_info['loaded']:
                print(f"❌ Model not loaded: {model_info.get('error', 'Unknown error')}")
                return jsonify({'error': 'Model not loaded. Please restart the application.'}), 500
            
            print(f"✅ Model loaded with {model_info['num_classes']} classes")
        except AttributeError:
            print("❌ get_model_info method not found in predictor")
            # Try basic check if model exists
            if not hasattr(breed_predictor, 'model') or breed_predictor.model is None:
                return jsonify({'error': 'Model not properly initialized'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"💾 File saved: {filepath}")
        
        # USE THE PREDICTOR CLASS FROM MODELS FOLDER - This is the key fix!
        prediction_result = breed_predictor.predict_breed(filepath)
        
        print(f"🎯 Prediction result from models/model_predictor.py: {prediction_result}")
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
            print(f"🗑️ Cleaned up file: {filepath}")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")
        
        # Check if prediction was successful
        if not prediction_result.get('success', True):
            return jsonify({
                'success': False,
                'error': prediction_result.get('error', 'Prediction failed'),
                'results': []
            }), 500

        # Format results for frontend (matching your existing JavaScript expectations)
        predictions = prediction_result.get('predictions', [])
        if not predictions and 'results' in prediction_result:
            predictions = prediction_result['results']
        
        formatted_results = []
        
        for i, pred in enumerate(predictions):
            # Handle different response formats from your predictor
            if isinstance(pred, dict):
                breed = pred.get('breed', pred.get('class_name', f'Class_{i}'))
                confidence = pred.get('confidence_percent', pred.get('confidence', 0) * 100)
                confidence_raw = pred.get('confidence', confidence / 100 if confidence > 1 else confidence)
            else:
                # Handle array format if that's what your predictor returns
                breed = f"Breed_{i}"
                confidence = float(pred) * 100 if pred <= 1 else float(pred)
                confidence_raw = float(pred) if pred <= 1 else float(pred) / 100
            
            formatted_results.append({
                'rank': i + 1,
                'breed': breed,
                'confidence': round(confidence, 1),
                'confidence_raw': confidence_raw
            })
        
        # Get confidence level and message from predictor if available
        confidence_level = prediction_result.get('confidence_level', 'medium')
        confidence_message = prediction_result.get('confidence_message', 'Analysis completed')
        
        # Return results in the format your JavaScript expects
        response_data = {
            'success': True,
            'message': 'Breed analysis completed successfully!',
            'results': formatted_results,
            'confidence_level': confidence_level,
            'confidence_message': confidence_message,
            'is_single_prediction': prediction_result.get('is_single_prediction', False)
        }
        
        print(f"📤 Sending response: {response_data}")
        return jsonify(response_data)
        
    except RequestEntityTooLarge:
        print("❌ File too large")
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        
    except Exception as e:
        print(f"❌ Error in upload_file: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        if hasattr(breed_predictor, 'get_model_info'):
            model_info = breed_predictor.get_model_info()
            return jsonify({
                'status': 'healthy',
                'model_loaded': model_info.get('loaded', False),
                'classes': model_info.get('num_classes', 0),
                'class_names': model_info.get('class_names', []),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'healthy',
                'model_loaded': hasattr(breed_predictor, 'model') and breed_predictor.model is not None,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify server is running"""
    return jsonify({
        'message': 'Server is running!',
        'endpoints': {
            '/': 'Home page',
            '/upload': 'File upload (POST)',
            '/predict': 'File prediction (POST)',
            '/health': 'Health check',
            '/debug/routes': 'List all routes'
        }
    })

@app.route('/model-info')
def model_info():
    """Get detailed model information"""
    try:
        if hasattr(breed_predictor, 'get_model_info'):
            info = breed_predictor.get_model_info()
        else:
            info = {
                'loaded': hasattr(breed_predictor, 'model') and breed_predictor.model is not None,
                'message': 'Basic predictor instance loaded'
            }
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'loaded': False,
            'error': str(e)
        })

# Add logging for all requests
@app.before_request
def log_request_info():
    print(f"🌐 {request.method} {request.path} - {request.remote_addr}")

@app.errorhandler(404)
def not_found(error):
    print(f"❌ 404 Error: {request.method} {request.path}")
    return jsonify({
        'error': 'Endpoint not found',
        'requested_path': request.path,
        'method': request.method,
        'available_endpoints': ['/', '/upload', '/predict', '/health', '/test', '/debug/routes']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ 500 Error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    print("=" * 60)
    print("🐄 Starting Cattle Breed Recognition System")
    print("=" * 60)
    
    # Check if predictor is properly loaded
    try:
        if hasattr(breed_predictor, 'get_model_info'):
            model_info = breed_predictor.get_model_info()
            if model_info.get('loaded', False):
                print("✅ System initialized successfully using models/model_predictor.py")
                if 'class_names' in model_info:
                    print(f"🧠 Model classes: {model_info['class_names']}")
                print("✅ Ready to analyze cattle breeds!")
            else:
                print("⚠️ System started but model may not be fully loaded")
                print(f"Model status: {model_info.get('error', 'Unknown status')}")
        else:
            print("✅ Basic predictor instance loaded")
            print("⚠️ get_model_info method not available - using basic validation")
        
        print(f"🌐 Server starting at http://127.0.0.1:5000")
        print("🔧 Debug endpoints available:")
        print("   - http://127.0.0.1:5000/test")
        print("   - http://127.0.0.1:5000/debug/routes")
        print("   - http://127.0.0.1:5000/health")
        print("   - http://127.0.0.1:5000/model-info")
        print("-" * 60)
        
        # Run Flask app
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=False
        )
    except Exception as e:
        print("❌ Failed to initialize system")
        print(f"Error: {e}")
        print("Please check your model files and models/model_predictor.py")
        traceback.print_exc()
        sys.exit(1)