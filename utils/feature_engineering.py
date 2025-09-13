import pandas as pd
import numpy as np
from config.pathogen_config import PATHOGENS
from config.mycotoxin_config import MYCOTOXINS

class FeatureEngineer:
    """Advanced feature engineering for wastewater surveillance data"""
    
    def __init__(self):
        self.created_features = []
    
    def create_features(self, df):
        """Create comprehensive feature set for ML models"""
        feature_df = df.copy()
        
        # Time-based features
        feature_df = self._create_temporal_features(feature_df)
        
        # Environmental interaction features
        feature_df = self._create_environmental_interactions(feature_df)
        
        # Microbiological ratio features
        feature_df = self._create_microbiological_ratios(feature_df)
        
        # Pathogen concentration features
        feature_df = self._create_pathogen_features(feature_df)
        
        # Mycotoxin features
        feature_df = self._create_mycotoxin_features(feature_df)
        
        # Resistance pattern features
        feature_df = self._create_resistance_features(feature_df)
        
        # Rolling window features
        feature_df = self._create_rolling_features(feature_df)
        
        # Categorical encoding
        feature_df = self._encode_categorical_features(feature_df)
        
        return feature_df
    
    def _create_temporal_features(self, df):
        """Create time-based cyclical features"""
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        if 'day_of_week' in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _create_environmental_interactions(self, df):
        """Create environmental parameter interaction features"""
        if 'temperature' in df.columns and 'ph_level' in df.columns:
            df['temp_ph_interaction'] = df['temperature'] * df['ph_level']
        
        if 'dissolved_oxygen' in df.columns and 'temperature' in df.columns:
            df['oxygen_temp_ratio'] = df['dissolved_oxygen'] / (df['temperature'] + 1)
        
        if 'turbidity' in df.columns and 'flow_rate' in df.columns:
            df['turbidity_flow_interaction'] = df['turbidity'] * df['flow_rate']
        
        if 'bod_5' in df.columns and 'cod' in df.columns:
            df['bod_cod_ratio'] = df['bod_5'] / (df['cod'] + 1)
        
        if 'total_nitrogen' in df.columns and 'total_phosphorus' in df.columns:
            df['nitrogen_phosphorus_ratio'] = df['total_nitrogen'] / (df['total_phosphorus'] + 1)
        
        # pH deviation from neutral
        if 'ph_level' in df.columns:
            df['ph_deviation'] = np.abs(df['ph_level'] - 7.0)
        
        # Temperature stress indicator
        if 'temperature' in df.columns:
            df['temp_stress'] = ((df['temperature'] < 10) | (df['temperature'] > 35)).astype(int)
        
        # Dissolved oxygen stress
        if 'dissolved_oxygen' in df.columns:
            df['oxygen_stress'] = (df['dissolved_oxygen'] < 4).astype(int)
        
        return df
    
    def _create_microbiological_ratios(self, df):
        """Create microbiological indicator ratios"""
        if 'fecal_coliforms' in df.columns and 'total_coliforms' in df.columns:
            df['fecal_total_coliform_ratio'] = df['fecal_coliforms'] / (df['total_coliforms'] + 1)
        
        if 'e_coli_indicator' in df.columns and 'fecal_coliforms' in df.columns:
            df['ecoli_fecal_ratio'] = df['e_coli_indicator'] / (df['fecal_coliforms'] + 1)
        
        # Log transformations for skewed distributions
        if 'total_coliforms' in df.columns:
            df['total_coliforms_log'] = np.log10(df['total_coliforms'] + 1)
        
        if 'fecal_coliforms' in df.columns:
            df['fecal_coliforms_log'] = np.log10(df['fecal_coliforms'] + 1)
        
        return df
    
    def _create_pathogen_features(self, df):
        """Create pathogen-related features"""
        # Log-transformed pathogen concentrations
        pathogen_concs = [col for col in df.columns if 'concentration' in col and 
                         any(pathogen in col for pathogen in PATHOGENS.keys())]
        
        for col in pathogen_concs:
            df[f'{col}_log'] = np.log10(df[col] + 1)
        
        # Pathogen diversity index (Shannon diversity)
        pathogen_detected_cols = [col for col in df.columns if 
                                col.endswith('_detected') and 
                                any(pathogen in col for pathogen in PATHOGENS.keys())]
        
        if pathogen_detected_cols:
            pathogen_matrix = df[pathogen_detected_cols].astype(int)
            df['pathogen_diversity'] = self._calculate_shannon_diversity(pathogen_matrix)
        
        # High-risk pathogen indicator
        high_risk_pathogens = [pathogen for pathogen, info in PATHOGENS.items() 
                              if info['risk_level'] == 'high']
        high_risk_cols = [f'{pathogen}_detected' for pathogen in high_risk_pathogens 
                         if f'{pathogen}_detected' in df.columns]
        
        if high_risk_cols:
            df['high_risk_pathogen_count'] = df[high_risk_cols].astype(int).sum(axis=1)
        
        return df
    
    def _create_mycotoxin_features(self, df):
        """Create mycotoxin-related features"""
        # Log-transformed mycotoxin concentrations
        mycotoxin_concs = [col for col in df.columns if 'concentration' in col and 
                          any(toxin.replace(' ', '_') in col for toxin in MYCOTOXINS.keys())]
        
        for col in mycotoxin_concs:
            df[f'{col}_log'] = np.log10(df[col] + 0.01)  # Small offset for zeros
        
        # Extremely high toxicity indicator
        extremely_high_toxins = [toxin for toxin, info in MYCOTOXINS.items() 
                               if info['toxicity'] == 'extremely_high']
        extremely_high_cols = [f'{toxin.replace(" ", "_")}_detected' for toxin in extremely_high_toxins 
                             if f'{toxin.replace(" ", "_")}_detected' in df.columns]
        
        if extremely_high_cols:
            df['extremely_high_toxin_count'] = df[extremely_high_cols].astype(int).sum(axis=1)
        
        # Mycotoxin source diversity
        source_groups = {}
        for toxin, info in MYCOTOXINS.items():
            source = info['source']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(f'{toxin.replace(" ", "_")}_detected')
        
        for source, toxin_cols in source_groups.items():
            available_cols = [col for col in toxin_cols if col in df.columns]
            if available_cols:
                df[f'{source.lower()}_mycotoxin_count'] = df[available_cols].astype(int).sum(axis=1)
        
        return df
    
    def _create_resistance_features(self, df):
        """Create antibiotic resistance features"""
        resistance_cols = ['ampicillin_resistance', 'tetracycline_resistance', 'multidrug_resistance']
        available_resistance_cols = [col for col in resistance_cols if col in df.columns]
        
        if available_resistance_cols:
            df['total_resistance_markers'] = df[available_resistance_cols].astype(int).sum(axis=1)
        
        return df
    
    def _create_rolling_features(self, df):
        """Create rolling window features for time series data"""
        if 'datetime' not in df.columns:
            return df
        
        df = df.sort_values('datetime')
        
        # 24-hour rolling features
        rolling_cols = ['temperature', 'ph_level', 'turbidity', 'total_coliforms', 'bod_5']
        available_rolling_cols = [col for col in rolling_cols if col in df.columns]
        
        for col in available_rolling_cols:
            # Rolling statistics
            df[f'{col}_24h_avg'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_24h_std'] = df[col].rolling(window=24, min_periods=1).std()
            df[f'{col}_24h_max'] = df[col].rolling(window=24, min_periods=1).max()
            df[f'{col}_24h_min'] = df[col].rolling(window=24, min_periods=1).min()
            
            # Deviation from rolling average
            df[f'{col}_deviation'] = df[col] - df[f'{col}_24h_avg']
            
            # Normalized deviation (Z-score)
            df[f'{col}_zscore'] = df[f'{col}_deviation'] / (df[f'{col}_24h_std'] + 1e-6)
        
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical features"""
        categorical_cols = ['location', 'facility_type', 'season']
        
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _calculate_shannon_diversity(self, binary_matrix):
        """Calculate Shannon diversity index for pathogen presence"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        proportions = binary_matrix.mean(axis=1) + epsilon
        
        # Calculate Shannon diversity
        shannon = -(proportions * np.log(proportions) + 
                   (1 - proportions) * np.log(1 - proportions))
        
        return shannon
    
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None