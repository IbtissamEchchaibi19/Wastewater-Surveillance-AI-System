# Pathogen reference data and classification system

PATHOGENS = {
    'Salmonella': {
        'family': 'Enterobacteriaceae', 
        'gram': 'negative', 
        'risk_level': 'high',
        'description': 'Major foodborne pathogen causing gastroenteritis',
        'typical_sources': ['poultry', 'eggs', 'produce', 'contaminated water'],
        'detection_limit': 1e3,  # copies/mL
        'infectious_dose': 1e2   # CFU
    },
    'E. coli': {
        'family': 'Enterobacteriaceae', 
        'gram': 'negative', 
        'risk_level': 'high',
        'description': 'Pathogenic strains cause severe gastrointestinal illness',
        'typical_sources': ['contaminated water', 'undercooked meat', 'fresh produce'],
        'detection_limit': 5e2,
        'infectious_dose': 1e1
    },
    'Campylobacter': {
        'family': 'Campylobacteraceae', 
        'gram': 'negative', 
        'risk_level': 'high',
        'description': 'Leading cause of bacterial gastroenteritis worldwide',
        'typical_sources': ['poultry', 'unpasteurized milk', 'contaminated water'],
        'detection_limit': 1e3,
        'infectious_dose': 5e2
    },
    'Listeria': {
        'family': 'Listeriaceae', 
        'gram': 'positive', 
        'risk_level': 'high',
        'description': 'Causes listeriosis, particularly dangerous for pregnant women',
        'typical_sources': ['deli meats', 'soft cheeses', 'ready-to-eat foods'],
        'detection_limit': 1e2,
        'infectious_dose': 1e6
    },
    'Shigella': {
        'family': 'Enterobacteriaceae', 
        'gram': 'negative', 
        'risk_level': 'high',
        'description': 'Causes shigellosis with severe diarrheal illness',
        'typical_sources': ['contaminated water', 'person-to-person contact'],
        'detection_limit': 1e3,
        'infectious_dose': 1e1
    },
    'Vibrio': {
        'family': 'Vibrionaceae', 
        'gram': 'negative', 
        'risk_level': 'medium',
        'description': 'Marine bacteria causing gastroenteritis and wound infections',
        'typical_sources': ['seafood', 'coastal waters'],
        'detection_limit': 1e3,
        'infectious_dose': 1e5
    },
    'Clostridium': {
        'family': 'Clostridiaceae', 
        'gram': 'positive', 
        'risk_level': 'medium',
        'description': 'Spore-forming bacteria producing potent toxins',
        'typical_sources': ['soil', 'sewage', 'food contamination'],
        'detection_limit': 1e2,
        'infectious_dose': 1e5
    },
    'Staphylococcus': {
        'family': 'Staphylococcaceae', 
        'gram': 'positive', 
        'risk_level': 'medium',
        'description': 'Produces enterotoxins causing food poisoning',
        'typical_sources': ['food handlers', 'dairy products', 'meat products'],
        'detection_limit': 1e3,
        'infectious_dose': 1e6
    },
    'Bacillus': {
        'family': 'Bacillaceae', 
        'gram': 'positive', 
        'risk_level': 'low',
        'description': 'Spore-forming bacteria, some species cause food poisoning',
        'typical_sources': ['soil', 'vegetation', 'contaminated foods'],
        'detection_limit': 1e3,
        'infectious_dose': 1e5
    },
    'Enterococcus': {
        'family': 'Enterococcaceae', 
        'gram': 'positive', 
        'risk_level': 'medium',
        'description': 'Indicator organism for fecal contamination',
        'typical_sources': ['human/animal feces', 'sewage'],
        'detection_limit': 1e2,
        'infectious_dose': 1e6
    }
}

# Risk level definitions
RISK_LEVEL_DEFINITIONS = {
    'high': {
        'description': 'Severe public health threat requiring immediate action',
        'response_time': '< 1 hour',
        'monitoring_frequency': 'Continuous',
        'notification_required': True
    },
    'medium': {
        'description': 'Moderate health concern requiring increased surveillance',
        'response_time': '< 4 hours',
        'monitoring_frequency': 'Every 2 hours',
        'notification_required': True
    },
    'low': {
        'description': 'Minimal health risk under normal monitoring',
        'response_time': '< 24 hours',
        'monitoring_frequency': 'Every 8 hours',
        'notification_required': False
    }
}

# Pathogen detection methods
DETECTION_METHODS = {
    'qPCR': {
        'description': 'Quantitative PCR for DNA/RNA detection',
        'sensitivity': 'High (1-10 copies)',
        'specificity': 'Very High (>99%)',
        'time_to_result': '2-4 hours'
    },
    'culture': {
        'description': 'Traditional culture-based identification',
        'sensitivity': 'Medium (100-1000 CFU)',
        'specificity': 'High (>95%)',
        'time_to_result': '24-72 hours'
    },
    'ELISA': {
        'description': 'Enzyme-linked immunosorbent assay',
        'sensitivity': 'Medium (1000-10000 cells)',
        'specificity': 'Medium (85-95%)',
        'time_to_result': '2-6 hours'
    },
    'next_gen_sequencing': {
        'description': 'Whole genome sequencing identification',
        'sensitivity': 'Very High (single cell)',
        'specificity': 'Very High (>99.9%)',
        'time_to_result': '4-24 hours'
    }
}

# Antibiotic resistance markers
RESISTANCE_MARKERS = {
    'ampicillin_resistance': {
        'gene': 'bla',
        'mechanism': 'Beta-lactamase production',
        'prevalence': 0.15
    },
    'tetracycline_resistance': {
        'gene': 'tetA/tetB',
        'mechanism': 'Efflux pump activation',
        'prevalence': 0.23
    },
    'multidrug_resistance': {
        'gene': 'Multiple',
        'mechanism': 'Various mechanisms',
        'prevalence': 0.08
    }
}