import json
from pathlib import Path
from config import Config

class LanguageSupport:
    def __init__(self):
        self.translations = {}
        self.load_translations()
    
    def load_translations(self):
        """Load translation files for all supported languages"""
        for lang in Config.SUPPORTED_LANGUAGES:
            translation_file = Config.TRANSLATIONS_DIR / f"{lang}.json"
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[lang] = json.load(f)
            else:
                self.translations[lang] = {}
    
    def get_text(self, key, language='en', **kwargs):
        """Get translated text for given key and language"""
        if language not in self.translations:
            language = 'en'
        
        text = self.translations[language].get(key, key)
        
        # Support for parameter substitution
        if kwargs:
            try:
                text = text.format(**kwargs)
            except:
                pass
        
        return text
    
    def get_available_languages(self):
        """Get list of available languages"""
        return Config.SUPPORTED_LANGUAGES

# Translation dictionaries
ENGLISH_TRANSLATIONS = {
    "app_title": "Breed Recognition System",
    "app_subtitle": "AI-Powered Cattle & Buffalo Breed Identification",
    "upload_title": "Upload Animal Image",
    "upload_instruction": "Select an image of cattle or buffalo for breed identification",
    "choose_file": "Choose File",
    "no_file_selected": "No file selected",
    "analyze_button": "Analyze Breed",
    "language_selector": "Language",
    "results_title": "Breed Identification Results",
    "confidence": "Confidence",
    "breed_information": "Breed Information",
    "characteristics": "Characteristics",
    "origin": "Origin",
    "avg_milk_yield": "Average Milk Yield",
    "body_weight": "Body Weight",
    "color": "Color",
    "male": "Male",
    "female": "Female",
    "back_button": "Upload Another Image",
    "error_no_file": "Please select an image file",
    "error_invalid_file": "Please upload a valid image file (PNG, JPG, JPEG, GIF)",
    "error_processing": "Error processing image. Please try again.",
    "prediction_single": "Identified Breed",
    "prediction_multiple": "Top Predicted Breeds",
    "cattle": "Cattle",
    "buffalo": "Buffalo",
    "supported_formats": "Supported formats: PNG, JPG, JPEG, GIF",
    "max_file_size": "Maximum file size: 16MB",
    "processing": "Processing image...",
    "upload_another": "Upload Another Image",
    "breed_details": "Breed Details",
    "milk_production": "Milk Production",
    "per_day": "per day",
    "kg": "kg",
    "liters": "liters",
    "indigenous": "Indigenous",
    "crossbred": "Crossbred",
    "dual_purpose": "Dual Purpose (Milk & Draft)",
    "milk_purpose": "Milk Purpose",
    "draft_purpose": "Draft Purpose",
    "loading": "Loading...",
    "try_again": "Try Again"
}

HINDI_TRANSLATIONS = {
    "app_title": "नस्ल पहचान प्रणाली",
    "app_subtitle": "AI-संचालित गोवंश और भैंस नस्ल पहचान",
    "upload_title": "पशु की छवि अपलोड करें",
    "upload_instruction": "नस्ल पहचान के लिए गोवंश या भैंस की छवि चुनें",
    "choose_file": "फ़ाइल चुनें",
    "no_file_selected": "कोई फ़ाइल नहीं चुनी गई",
    "analyze_button": "नस्ल का विश्लेषण करें",
    "language_selector": "भाषा",
    "results_title": "नस्ल पहचान परिणाम",
    "confidence": "विश्वसनीयता",
    "breed_information": "नस्ल की जानकारी",
    "characteristics": "विशेषताएं",
    "origin": "मूल स्थान",
    "avg_milk_yield": "औसत दूध उत्पादन",
    "body_weight": "शरीर का वजन",
    "color": "रंग",
    "male": "नर",
    "female": "मादा",
    "back_button": "दूसरी छवि अपलोड करें",
    "error_no_file": "कृपया एक छवि फ़ाइल चुनें",
    "error_invalid_file": "कृपया एक वैध छवि फ़ाइल अपलोड करें (PNG, JPG, JPEG, GIF)",
    "error_processing": "छवि प्रसंस्करण में त्रुटि। कृपया पुनः प्रयास करें।",
    "prediction_single": "पहचानी गई नस्ल",
    "prediction_multiple": "शीर्ष भविष्यवाणी नस्लें",
    "cattle": "गोवंश",
    "buffalo": "भैंस",
    "supported_formats": "समर्थित प्रारूप: PNG, JPG, JPEG, GIF",
    "max_file_size": "अधिकतम फ़ाइल आकार: 16MB",
    "processing": "छवि प्रसंस्करण हो रहा है...",
    "upload_another": "दूसरी छवि अपलोड करें",
    "breed_details": "नस्ल विवरण",
    "milk_production": "दूध उत्पादन",
    "per_day": "प्रति दिन",
    "kg": "किग्रा",
    "liters": "लीटर",
    "indigenous": "देशी",
    "crossbred": "संकर",
    "dual_purpose": "द्विगुणी उपयोग (दूध और मसौदा)",
    "milk_purpose": "दूध उद्देश्य",
    "draft_purpose": "मसौदा उद्देश्य",
    "loading": "लोड हो रहा है...",
    "try_again": "पुनः प्रयास करें"
}

TAMIL_TRANSLATIONS = {
    "app_title": "இன அடையாள அமைப்பு",
    "app_subtitle": "AI-இயங்கும் கால்நடை மற்றும் எருமை இன அடையாளம்",
    "upload_title": "விலங்கின் படத்தைப் பதிவேற்றவும்",
    "upload_instruction": "இன அடையாளத்திற்கு கால்நடை அல்லது எருமையின் படத்தைத் தேர்ந்தெடுக்கவும்",
    "choose_file": "கோப்பைத் தேர்ந்தெடுக்கவும்",
    "no_file_selected": "எந்த கோப்பும் தேர்ந்தெடுக்கப்படவில்லை",
    "analyze_button": "இனத்தை பகுப்பாய்வு செய்யவும்",
    "language_selector": "மொழி",
    "results_title": "இன அடையாள முடிவுகள்",
    "confidence": "நம்பகத்தன்மை",
    "breed_information": "இன தகவல்",
    "characteristics": "சிறப்பியல்புகள்",
    "origin": "தோற்றம்",
    "avg_milk_yield": "சராசரி பால் உற்பத்தி",
    "body_weight": "உடல் எடை",
    "color": "நிறம்",
    "male": "ஆண்",
    "female": "பெண்",
    "back_button": "மற்றொரு படத்தைப் பதிவேற்றவும்",
    "error_no_file": "தயவுசெய்து ஒரு படக் கோப்பைத் தேர்ந்தெடுக்கவும்",
    "error_invalid_file": "தயவுசெய்து சரியான படக் கோப்பைப் பதிவேற்றவும் (PNG, JPG, JPEG, GIF)",
    "error_processing": "படத்தை செயலாக்குவதில் பிழை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
    "prediction_single": "அடையாளம் காணப்பட்ட இனம்",
    "prediction_multiple": "முதன்மை கணிக்கப்பட்ட இனங்கள்",
    "cattle": "கால்நடை",
    "buffalo": "எருமை",
    "supported_formats": "ஆதரிக்கப்படும் வடிவங்கள்: PNG, JPG, JPEG, GIF",
    "max_file_size": "அதிகபட்ச கோப்பு அளவு: 16MB",
    "processing": "படம் செயலாக்கப்படுகிறது...",
    "upload_another": "மற்றொரு படத்தைப் பதிவேற்றவும்",
    "breed_details": "இன விவரங்கள்",
    "milk_production": "பால் உற்பத்தி",
    "per_day": "ஒரு நாளுக்கு",
    "kg": "கிலோ",
    "liters": "லிட்டர்",
    "indigenous": "உள்நாட்டு",
    "crossbred": "கலப்பு",
    "dual_purpose": "இரட்டை நோக்கம் (பால் மற்றும் இழுத்தல்)",
    "milk_purpose": "பால் நோக்கம்",
    "draft_purpose": "இழுத்தல் நோக்கம்",
    "loading": "ஏற்றுகிறது...",
    "try_again": "மீண்டும் முயற்சிக்கவும்"
}

def create_translation_files():
    """Create translation JSON files"""
    translations = {
        'en': ENGLISH_TRANSLATIONS,
        'hi': HINDI_TRANSLATIONS,
        'ta': TAMIL_TRANSLATIONS
    }
    
    # Create translations directory if it doesn't exist
    Config.TRANSLATIONS_DIR.mkdir(exist_ok=True)
    
    for lang, content in translations.items():
        file_path = Config.TRANSLATIONS_DIR / f"{lang}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        print(f"Created translation file: {file_path}")

# Initialize language support
def initialize_language_support():
    """Initialize language support by creating translation files"""
    create_translation_files()
    return LanguageSupport()

# Create instance
language_support = initialize_language_support()

# Language mapping for display
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'हिन्दी (Hindi)',
    'ta': 'தமிழ் (Tamil)'
}

def get_language_name(lang_code):
    """Get display name for language code"""
    return LANGUAGE_NAMES.get(lang_code, lang_code)