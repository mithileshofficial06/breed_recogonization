"""
Breed Database containing information about cattle and buffalo breeds
"""

BREED_INFO = {
    # Cattle Breeds
    'Gir': {
        'type': 'cattle',
        'origin': 'Gujarat, India',
        'characteristics': {
            'en': 'Known for high milk yield and disease resistance. Has distinctive curved horns and compact body.',
            'hi': 'उच्च दूध उत्पादन और रोग प्रतिरोधक क्षमता के लिए प्रसिद्ध। विशिष्ट घुमावदार सींग और कॉम्पैक्ट शरीर।',
            'ta': 'அதிக பால் உற்பத்தி மற்றும் நோய் எதிர்ப்பு சக்தி கொண்டது. தனித்துவமான வளைந்த கொம்புகள் மற்றும் கச்சிதமான உடல்.'
        },
        'avg_milk_yield': {
            'en': '1,500-2,500 liters per lactation',
            'hi': '1,500-2,500 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 1,500-2,500 லிட்டர்'
        },
        'body_weight': {
            'male': '450-550 kg',
            'female': '300-400 kg'
        },
        'color': {
            'en': 'White to light grey with red/brown patches',
            'hi': 'लाल/भूरे धब्बों के साथ सफेद से हल्का भूरा',
            'ta': 'சிவப்பு/பழுப்பு புள்ளிகளுடன் வெள்ளை முதல் வெளிர் சாம்பல்'
        }
    },
    
    'Alamabadi': {
        'type': 'cattle',
        'origin': 'Tamil Nadu, India',
        'characteristics': {
            'en': 'Dual purpose breed good for milk and draft work. Hardy and well-adapted to local conditions.',
            'hi': 'दूध और खेतीबाड़ी के काम के लिए उपयुक्त द्विउद्देश्यीय नस्ल। कठोर और स्थानीय परिस्थितियों के अनुकूल।',
            'ta': 'பால் மற்றும் விவசாய வேலைக்கு ஏற்ற இரட்டை நோக்க இனம். கடினமான மற்றும் உள்ளூர் சூழ்நிலைகளுக்கு ஏற்ற.'
        },
        'avg_milk_yield': {
            'en': '800-1,200 liters per lactation',
            'hi': '800-1,200 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 800-1,200 லிட்டர்'
        },
        'body_weight': {
            'male': '400-500 kg',
            'female': '250-350 kg'
        },
        'color': {
            'en': 'Grey to dark grey with black markings',
            'hi': 'काले निशान के साथ भूरा से गहरा भूरा',
            'ta': 'கருப்பு குறிகளுடன் சாம்பல் முதல் அடர் சாம்பல்'
        }
    },
    
    'Bargur': {
        'type': 'cattle',
        'origin': 'Tamil Nadu, India',
        'characteristics': {
            'en': 'Small-sized hardy breed, excellent for hilly terrain. Known for disease resistance.',
            'hi': 'छोटे आकार की कठोर नस्ल, पहाड़ी इलाकों के लिए उत्कृष्ट। रोग प्रतिरोधक क्षमता के लिए जानी जाती है।',
            'ta': 'சிறிய அளவிலான கடினமான இனம், மலைப்பாங்கான பகுதிகளுக்கு சிறந்தது. நோய் எதிர்ப்பு சக்திக்கு பெயர் பெற்றது.'
        },
        'avg_milk_yield': {
            'en': '400-800 liters per lactation',
            'hi': '400-800 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 400-800 லிட்டர்'
        },
        'body_weight': {
            'male': '250-350 kg',
            'female': '150-250 kg'
        },
        'color': {
            'en': 'Grey with white markings',
            'hi': 'सफेद निशान के साथ भूरा',
            'ta': 'வெள்ளை குறிகளுடன் சாம்பல்'
        }
    },
    
    'Kangayam': {
        'type': 'cattle',
        'origin': 'Tamil Nadu, India',
        'characteristics': {
            'en': 'Powerful draft breed with excellent working ability. Good milk production.',
            'hi': 'उत्कृष्ट कार्य क्षमता के साथ शक्तिशाली मवेशी नस्ल। अच्छा दूध उत्पादन।',
            'ta': 'சிறந்த வேலை திறனுடன் வலிமையான கர்ம இனம். நல்ல பால் உற்பத்தி.'
        },
        'avg_milk_yield': {
            'en': '600-1,000 liters per lactation',
            'hi': '600-1,000 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 600-1,000 லிட்டர்'
        },
        'body_weight': {
            'male': '450-550 kg',
            'female': '300-400 kg'
        },
        'color': {
            'en': 'Red with white markings',
            'hi': 'सफेद निशान के साथ लाल',
            'ta': 'வெள்ளை குறிகளுடன் சிவப்பு'
        }
    },
    
    'Pulikulam': {
        'type': 'cattle',
        'origin': 'Tamil Nadu, India',
        'characteristics': {
            'en': 'Known for speed and agility, used in traditional bull racing. Compact and sturdy.',
            'hi': 'गति और फुर्ती के लिए जानी जाती है, पारंपरिक बैल दौड़ में प्रयुक्त। कॉम्पैक्ट और मजबूत।',
            'ta': 'வேகம் மற்றும் சுறுசுறுப்புக்கு பெயர் பெற்றது, பாரம்பரிய காளை ஓட்டத்தில் பயன்படுத்தப்படுகிறது. கச்சிதமான மற்றும் உறுதியான.'
        },
        'avg_milk_yield': {
            'en': '300-600 liters per lactation',
            'hi': '300-600 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 300-600 லிட்டர்'
        },
        'body_weight': {
            'male': '300-400 kg',
            'female': '200-300 kg'
        },
        'color': {
            'en': 'Light grey to white',
            'hi': 'हल्का भूरा से सफेद',
            'ta': 'வெளிர் சாம்பல் முதல் வெள்ளை'
        }
    },
    
    'Umblachery': {
        'type': 'cattle',
        'origin': 'Tamil Nadu, India',
        'characteristics': {
            'en': 'Draught breed with good endurance. Well adapted to coastal regions.',
            'hi': 'अच्छी सहनशीलता के साथ मसौदा नस्ल। तटीय क्षेत्रों के अनुकूल।',
            'ta': 'நல்ல சகிப்புத்தன்மையுடன் கூடிய இழுக்கும் இனம். கடலோர பகுதிகளுக்கு நன்கு ஏற்றது.'
        },
        'avg_milk_yield': {
            'en': '500-900 liters per lactation',
            'hi': '500-900 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 500-900 லிட்டர்'
        },
        'body_weight': {
            'male': '350-450 kg',
            'female': '250-350 kg'
        },
        'color': {
            'en': 'Grey to dark grey',
            'hi': 'भूरा से गहरा भूरा',
            'ta': 'சாம்பல் முதல் அடர் சாம்பல்'
        }
    },
    
    # Buffalo Breeds
    'Murrah': {
        'type': 'buffalo',
        'origin': 'Haryana, India',
        'characteristics': {
            'en': 'World-famous buffalo breed with highest milk yield. Large size with curved horns.',
            'hi': 'सबसे अधिक दूध उत्पादन के साथ विश्व प्रसिद्ध भैंस नस्ल। घुमावदार सींग के साथ बड़ा आकार।',
            'ta': 'அதிக பால் உற்பத்தியுடன் உலகப் பிரசித்தி பெற்ற எருமை இனம். வளைந்த கொம்புகளுடன் பெரிய அளவு.'
        },
        'avg_milk_yield': {
            'en': '2,500-4,000 liters per lactation',
            'hi': '2,500-4,000 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 2,500-4,000 லிட்டர்'
        },
        'body_weight': {
            'male': '550-800 kg',
            'female': '450-650 kg'
        },
        'color': {
            'en': 'Jet black with white markings on face and legs',
            'hi': 'चेहरे और पैरों पर सफेद निशान के साथ जेट काला',
            'ta': 'முகம் மற்றும் கால்களில் வெள்ளை குறிகளுடன் கரும்நிறம்'
        }
    },
    
    'Toda': {
        'type': 'buffalo',
        'origin': 'Tamil Nadu, India (Nilgiris)',
        'characteristics': {
            'en': 'Indigenous buffalo breed of Nilgiris, adapted to hilly terrain. Medium-sized with good milk production.',
            'hi': 'नीलगिरि की स्वदेशी भैंस नस्ल, पहाड़ी इलाकों के अनुकूल। अच्छे दूध उत्पादन के साथ मध्यम आकार।',
            'ta': 'நீலகிரியின் பழங்குடி எருமை இனம், மலைப்பாங்கான பகுதிகளுக்கு ஏற்றது. நல்ல பால் உற்பத்தியுடன் நடுத்தர அளவு.'
        },
        'avg_milk_yield': {
            'en': '1,200-2,000 liters per lactation',
            'hi': '1,200-2,000 लीटर प्रति स्तनपान काल',
            'ta': 'ஒரு பால் கறக்கும் காலத்தில் 1,200-2,000 லிட்டர்'
        },
        'body_weight': {
            'male': '400-500 kg',
            'female': '300-400 kg'
        },
        'color': {
            'en': 'Black to dark brown',
            'hi': 'काला से गहरा भूरा',
            'ta': 'கருப்பு முதல் அடர் பழுப்பு'
        }
    }
}

def get_breed_info(breed_name, language='en'):
    """Get breed information in specified language"""
    breed = BREED_INFO.get(breed_name)
    if not breed:
        return None
    
    return {
        'name': breed_name,
        'type': breed['type'],
        'origin': breed['origin'],
        'characteristics': breed['characteristics'].get(language, breed['characteristics']['en']),
        'avg_milk_yield': breed['avg_milk_yield'].get(language, breed['avg_milk_yield']['en']),
        'body_weight': breed['body_weight'],
        'color': breed['color'].get(language, breed['color']['en'])
    }

def get_all_breeds():
    """Get list of all supported breeds"""
    return list(BREED_INFO.keys())