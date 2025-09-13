# Configuration settings for the Wastewater Surveillance AI system

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Wastewater Surveillance AI",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Dataset generation configuration
DATASET_CONFIG = {
    "default_size": 15000,
    "size_options": [10000, 15000, 25000],
    "default_time_period": 180,
    "min_time_period": 30,
    "max_time_period": 365,
    "default_threshold": 100,
    "min_threshold": 50,
    "max_threshold": 150
}

# Model training configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.1,
    "contamination_rate": 0.1
}

# Environmental parameter ranges
ENVIRONMENTAL_RANGES = {
    "temperature": {"min": 0.0, "max": 40.0, "default": 25.0},
    "ph_level": {"min": 5.0, "max": 9.0, "default": 7.0},
    "dissolved_oxygen": {"min": 0.0, "max": 15.0, "default": 8.0},
    "turbidity": {"min": 0.0, "max": 50.0, "default": 10.0},
    "conductivity": {"min": 100, "max": 2000, "default": 800},
    "flow_rate": {"min": 0.1, "max": 100.0, "default": 10.0},
    "bod_5": {"min": 0.0, "max": 200.0, "default": 30.0},
    "cod": {"min": 0.0, "max": 500.0, "default": 80.0},
    "total_nitrogen": {"min": 0.0, "max": 50.0, "default": 15.0},
    "total_phosphorus": {"min": 0.0, "max": 20.0, "default": 5.0}
}

# Microbiological parameter ranges
MICROBIOLOGICAL_RANGES = {
    "total_coliforms": {"min": 0, "max": 100000, "default": 10000},
    "fecal_coliforms": {"min": 0, "max": 50000, "default": 2000},
    "e_coli_indicator": {"min": 0, "max": 20000, "default": 1000}
}

# Location and facility options
LOCATION_OPTIONS = [
    'Treatment_Plant_A', 'Treatment_Plant_B', 'Industrial_Zone_C', 
    'Residential_Area_D', 'Agricultural_District_E', 'Commercial_Hub_F'
]

FACILITY_TYPE_OPTIONS = [
    'Primary', 'Secondary', 'Tertiary', 'Industrial', 'Agricultural'
]

# Risk assessment thresholds
RISK_THRESHOLDS = {
    "critical": 100,
    "high": 60,
    "medium": 30,
    "low": 0
}

# Color schemes for visualizations
COLOR_SCHEMES = {
    "risk_colors": {
        'Critical': '#d63031', 
        'High': '#e17055', 
        'Medium': '#fdcb6e', 
        'Low': '#00b894'
    },
    "pathogen_colors": {
        'high': '#d63031', 
        'medium': '#e17055', 
        'low': '#00b894'
    },
    "toxicity_colors": {
        'Extremely High': '#d63031', 
        'High': '#e17055', 
        'Medium': '#fdcb6e', 
        'Low': '#00b894'
    }
}