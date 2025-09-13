
MYCOTOXINS = {
    'Aflatoxin B1': {
        'source': 'Aspergillus',
        'toxicity': 'extremely_high',
        'stability': 'high',
        'molecular_weight': 312.27,
        'cas_number': '1162-65-8',
        'detection_limit': 0.01,  
        'regulatory_limit': 0.1,   
        'health_effects': ['hepatotoxic', 'carcinogenic', 'mutagenic'],
        'target_organs': ['liver', 'kidney']
    },
    'Aflatoxin B2': {
        'source': 'Aspergillus',
        'toxicity': 'high',
        'stability': 'high',
        'molecular_weight': 314.29,
        'cas_number': '7220-81-7',
        'detection_limit': 0.01,
        'regulatory_limit': 0.1,
        'health_effects': ['hepatotoxic', 'carcinogenic'],
        'target_organs': ['liver']
    },
    'Ochratoxin A': {
        'source': 'Penicillium/Aspergillus',
        'toxicity': 'high',
        'stability': 'medium',
        'molecular_weight': 403.81,
        'cas_number': '303-47-9',
        'detection_limit': 0.02,
        'regulatory_limit': 0.5,
        'health_effects': ['nephrotoxic', 'carcinogenic', 'immunosuppressive'],
        'target_organs': ['kidney', 'liver']
    },
    'Deoxynivalenol': {
        'source': 'Fusarium',
        'toxicity': 'medium',
        'stability': 'medium',
        'molecular_weight': 296.32,
        'cas_number': '51481-10-8',
        'detection_limit': 0.05,
        'regulatory_limit': 1.0,
        'health_effects': ['gastrointestinal', 'immunosuppressive'],
        'target_organs': ['gastrointestinal_tract', 'immune_system']
    },
    'Zearalenone': {
        'source': 'Fusarium',
        'toxicity': 'medium',
        'stability': 'low',
        'molecular_weight': 318.36,
        'cas_number': '17924-92-4',
        'detection_limit': 0.1,
        'regulatory_limit': 2.0,
        'health_effects': ['estrogenic', 'reproductive_toxicity'],
        'target_organs': ['reproductive_system']
    },
    'T-2 Toxin': {
        'source': 'Fusarium',
        'toxicity': 'high',
        'stability': 'medium',
        'molecular_weight': 466.52,
        'cas_number': '21259-20-1',
        'detection_limit': 0.02,
        'regulatory_limit': 0.2,
        'health_effects': ['cytotoxic', 'immunosuppressive', 'dermatotoxic'],
        'target_organs': ['skin', 'immune_system', 'gastrointestinal_tract']
    },
    'Fumonisin B1': {
        'source': 'Fusarium',
        'toxicity': 'medium',
        'stability': 'high',
        'molecular_weight': 721.83,
        'cas_number': '116355-83-0',
        'detection_limit': 0.1,
        'regulatory_limit': 4.0,
        'health_effects': ['hepatotoxic', 'nephrotoxic'],
        'target_organs': ['liver', 'kidney']
    },
    'Patulin': {
        'source': 'Penicillium',
        'toxicity': 'medium',
        'stability': 'low',
        'molecular_weight': 154.12,
        'cas_number': '149-29-1',
        'detection_limit': 0.05,
        'regulatory_limit': 1.0,
        'health_effects': ['gastrointestinal', 'immunotoxic'],
        'target_organs': ['gastrointestinal_tract']
    },
    'Citrinin': {
        'source': 'Penicillium',
        'toxicity': 'medium',
        'stability': 'medium',
        'molecular_weight': 250.25,
        'cas_number': '518-75-2',
        'detection_limit': 0.02,
        'regulatory_limit': 0.5,
        'health_effects': ['nephrotoxic', 'hepatotoxic'],
        'target_organs': ['kidney', 'liver']
    },
    'Ergot Alkaloids': {
        'source': 'Claviceps',
        'toxicity': 'high',
        'stability': 'medium',
        'molecular_weight': 325.40,
        'cas_number': '113-15-5',
        'detection_limit': 0.01,
        'regulatory_limit': 0.1,
        'health_effects': ['neurotoxic', 'vasoconstrictive'],
        'target_organs': ['nervous_system', 'cardiovascular_system']
    }
}

# Toxicity level definitions
TOXICITY_LEVELS = {
    'extremely_high': {
        'description': 'Immediate severe health threat',
        'action_level': 0.01,  # ng/L
        'response_time': 'Immediate',
        'public_notification': True,
        'treatment_required': True
    },
    'high': {
        'description': 'Significant health risk',
        'action_level': 0.05,
        'response_time': '< 2 hours',
        'public_notification': True,
        'treatment_required': True
    },
    'medium': {
        'description': 'Moderate health concern',
        'action_level': 0.1,
        'response_time': '< 8 hours',
        'public_notification': False,
        'treatment_required': False
    },
    'low': {
        'description': 'Minimal health impact',
        'action_level': 1.0,
        'response_time': '< 24 hours',
        'public_notification': False,
        'treatment_required': False
    }
}

# Fungal source information
FUNGAL_SOURCES = {
    'Aspergillus': {
        'species': ['A. flavus', 'A. parasiticus', 'A. niger'],
        'optimal_conditions': {
            'temperature': '25-35°C',
            'humidity': '>80%',
            'ph': '6.0-8.0'
        },
        'substrates': ['grains', 'nuts', 'seeds', 'organic_matter'],
        'mycotoxins_produced': ['Aflatoxin B1', 'Aflatoxin B2', 'Ochratoxin A']
    },
    'Fusarium': {
        'species': ['F. graminearum', 'F. culmorum', 'F. verticillioides'],
        'optimal_conditions': {
            'temperature': '20-28°C',
            'humidity': '>70%',
            'ph': '5.0-7.0'
        },
        'substrates': ['cereals', 'corn', 'wheat', 'plant_debris'],
        'mycotoxins_produced': ['Deoxynivalenol', 'Zearalenone', 'T-2 Toxin', 'Fumonisin B1']
    },
    'Penicillium': {
        'species': ['P. verrucosum', 'P. expansum', 'P. citrinum'],
        'optimal_conditions': {
            'temperature': '15-25°C',
            'humidity': '>75%',
            'ph': '4.0-7.0'
        },
        'substrates': ['fruits', 'vegetables', 'stored_grains'],
        'mycotoxins_produced': ['Ochratoxin A', 'Patulin', 'Citrinin']
    },
    'Claviceps': {
        'species': ['C. purpurea', 'C. fusiformis'],
        'optimal_conditions': {
            'temperature': '18-24°C',
            'humidity': '>90%',
            'ph': '5.5-7.5'
        },
        'substrates': ['rye', 'wheat', 'barley', 'grasses'],
        'mycotoxins_produced': ['Ergot Alkaloids']
    }
}

# Detection and analysis methods
DETECTION_METHODS = {
    'HPLC-MS/MS': {
        'description': 'High Performance Liquid Chromatography - Tandem Mass Spectrometry',
        'sensitivity': 'Very High (0.01-0.1 ng/L)',
        'specificity': 'Very High (>99%)',
        'cost': 'High',
        'analysis_time': '30-60 minutes'
    },
    'ELISA': {
        'description': 'Enzyme-Linked Immunosorbent Assay',
        'sensitivity': 'Medium (0.1-1.0 ng/L)',
        'specificity': 'High (>95%)',
        'cost': 'Medium',
        'analysis_time': '2-4 hours'
    },
    'LC-MS': {
        'description': 'Liquid Chromatography - Mass Spectrometry',
        'sensitivity': 'High (0.05-0.5 ng/L)',
        'specificity': 'High (>98%)',
        'cost': 'High',
        'analysis_time': '45-90 minutes'
    }
}