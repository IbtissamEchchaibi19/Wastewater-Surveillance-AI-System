import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from config.settings import PAGE_CONFIG, DATASET_CONFIG
from config.pathogen_config import PATHOGENS
from config.mycotoxin_config import MYCOTOXINS
from data.data_generator import generate_wastewater_surveillance_data
from models.surveillance_ai import WastewaterSurveillanceAI
from dashboard.styling import load_custom_css
from dashboard.dashboard_components import (
    render_header,
    render_sidebar_config,
    render_surveillance_tab,
    render_model_training_tab,
    render_pathogen_analysis_tab,
    render_mycotoxin_analysis_tab,
    render_outbreak_detection_tab,
    render_risk_prediction_tab
)

# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# Load custom CSS styling
load_custom_css()

# Main application header
render_header()

# Sidebar configuration
dataset_size, time_period_days, contamination_threshold = render_sidebar_config()

# Generate sophisticated dataset
@st.cache_data
def load_surveillance_data(n_samples, time_days):
    """Load and cache surveillance data"""
    return generate_wastewater_surveillance_data(n_samples, time_days)

with st.spinner("Generating advanced wastewater surveillance dataset..."):
    df = load_surveillance_data(dataset_size, time_period_days)

# Initialize AI system
if 'surveillance_ai' not in st.session_state:
    st.session_state.surveillance_ai = WastewaterSurveillanceAI()

# Create main navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Surveillance Dashboard",
    "🧠 AI Model Training", 
    "🔬 Pathogen Analysis",
    "☢️ Mycotoxin Detection",
    "⚠️ Outbreak Detection",
    "🎯 Risk Prediction"
])

with tab1:
    render_surveillance_tab(df, contamination_threshold)

with tab2:
    render_model_training_tab(st.session_state.surveillance_ai, df)

with tab3:
    render_pathogen_analysis_tab(df, PATHOGENS)

with tab4:
    render_mycotoxin_analysis_tab(df, MYCOTOXINS)

with tab5:
    render_outbreak_detection_tab(df)

with tab6:
    render_risk_prediction_tab(st.session_state.surveillance_ai, df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <strong>🔬 Advanced Wastewater Pathogen & Mycotoxin Surveillance AI</strong><br>
    Real-time monitoring • Machine learning prediction • Early outbreak detection<br>
    <em>Protecting public health through intelligent wastewater analysis</em>
</div>
""", unsafe_allow_html=True)