import pandas as pd
import numpy as np
from sklearn.ensemble import (IsolationForest, RandomForestRegressor, RandomForestClassifier,
                            GradientBoostingRegressor, GradientBoostingClassifier,
                            ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, 
                                LogisticRegression, SGDClassifier)
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                 PowerTransformer, QuantileTransformer)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           precision_score, recall_score, f1_score)
from sklearn.covariance import EllipticEnvelope
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports (conditional)
try:
    import xgboost as xgb
    from lightgbm import LGBMRegressor, LGBMClassifier
    ADVANCED_LIBS = True
except ImportError:
    ADVANCED_LIBS = False

from utils.feature_engineering import FeatureEngineer
from config.pathogen_config import PATHOGENS
from config.mycotoxin_config import MYCOTOXINS

class WastewaterSurveillanceAI:
    """Advanced AI system for wastewater pathogen and mycotoxin surveillance"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.anomaly_detectors = {}
        self.time_series_models = {}
        self.feature_columns = []
        self.feature_engineer = FeatureEngineer()
        
    def prepare_features(self, df):
        """Advanced feature engineering for wastewater analysis"""
        return self.feature_engineer.create_features(df)
    
    def train_pathogen_detection_models(self, df):
        """Train models for pathogen detection and classification"""
        X = self.prepare_features(df)
        
        # Select numeric features for modeling
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not any(
            target in col.lower() for target in ['risk', 'detected', 'concentration', 'outbreak']
        )]
        
        X_features = X[feature_cols]
        
        # Remove columns with too many NaN values
        X_features = X_features.dropna(axis=1, thresh=len(X_features) * 0.5)
        X_features = X_features.fillna(X_features.median())
        
        self.feature_columns = X_features.columns.tolist()
        
        results = {}
        
        # 1. Pathogen Risk Score Prediction (Regression)
        y_pathogen_risk = df['pathogen_risk_score']
        
        pathogen_models = self._get_regression_models()
        
        # Split data with time-aware split
        if 'datetime' in df.columns:
            split_idx = int(len(X_features) * 0.8)
            X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
            y_train, y_test = y_pathogen_risk.iloc[:split_idx], y_pathogen_risk.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_pathogen_risk, test_size=0.2, random_state=42
            )
        
        # Scale features
        self.scalers['pathogen'] = StandardScaler()
        X_train_scaled = self.scalers['pathogen'].fit_transform(X_train)
        X_test_scaled = self.scalers['pathogen'].transform(X_test)
        
        # Train pathogen models
        for name, model in pathogen_models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                results[f'pathogen_{name}'] = {
                    'model': model,
                    'r2_score': r2,
                    'mse': mse,
                    'type': 'regression'
                }
                
                if 'pathogen_best' not in results or r2 > results['pathogen_best']['r2_score']:
                    results['pathogen_best'] = results[f'pathogen_{name}'].copy()
                    results['pathogen_best']['name'] = name
                    
            except Exception as e:
                print(f"Warning: Error training pathogen model {name}: {str(e)}")
        
        # 2. Mycotoxin Risk Score Prediction
        y_mycotoxin_risk = df['mycotoxin_risk_score']
        mycotoxin_models = self._get_regression_models()
        
        if 'datetime' in df.columns:
            y_myco_train, y_myco_test = y_mycotoxin_risk.iloc[:split_idx], y_mycotoxin_risk.iloc[split_idx:]
        else:
            _, _, y_myco_train, y_myco_test = train_test_split(
                X_features, y_mycotoxin_risk, test_size=0.2, random_state=42
            )
        
        for name, model in mycotoxin_models.items():
            try:
                model.fit(X_train_scaled, y_myco_train)
                y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_myco_test, y_pred)
                mse = mean_squared_error(y_myco_test, y_pred)
                
                results[f'mycotoxin_{name}'] = {
                    'model': model,
                    'r2_score': r2,
                    'mse': mse,
                    'type': 'regression'
                }
                
                if 'mycotoxin_best' not in results or r2 > results['mycotoxin_best']['r2_score']:
                    results['mycotoxin_best'] = results[f'mycotoxin_{name}'].copy()
                    results['mycotoxin_best']['name'] = name
                    
            except Exception as e:
                print(f"Warning: Error training mycotoxin model {name}: {str(e)}")
        
        # 3. Risk Classification
        y_risk_category = df['risk_category']
        classification_models = self._get_classification_models()
        
        if 'datetime' in df.columns:
            y_class_train, y_class_test = y_risk_category.iloc[:split_idx], y_risk_category.iloc[split_idx:]
        else:
            _, _, y_class_train, y_class_test = train_test_split(
                X_features, y_risk_category, test_size=0.2, random_state=42, 
                stratify=y_risk_category
            )
        
        for name, model in classification_models.items():
            try:
                model.fit(X_train_scaled, y_class_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_class_test, y_pred)
                precision = precision_score(y_class_test, y_pred, average='weighted')
                recall = recall_score(y_class_test, y_pred, average='weighted')
                f1 = f1_score(y_class_test, y_pred, average='weighted')
                
                results[f'classification_{name}'] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'type': 'classification'
                }
                
                if 'classification_best' not in results or accuracy > results['classification_best']['accuracy']:
                    results['classification_best'] = results[f'classification_{name}'].copy()
                    results['classification_best']['name'] = name
                    
            except Exception as e:
                print(f"Warning: Error training classification model {name}: {str(e)}")
        
        # 4. Anomaly Detection
        anomaly_models = self._get_anomaly_models()
        
        for name, model in anomaly_models.items():
            try:
                if name == 'Local Outlier Factor':
                    anomaly_scores = model.fit_predict(X_train_scaled)
                else:
                    model.fit(X_train_scaled)
                    anomaly_scores = model.predict(X_test_scaled)
                
                anomalies_detected = len([score for score in anomaly_scores if score == -1])
                anomaly_rate = anomalies_detected / len(anomaly_scores) if len(anomaly_scores) > 0 else 0
                
                results[f'anomaly_{name}'] = {
                    'model': model,
                    'anomaly_rate': anomaly_rate,
                    'anomalies_detected': anomalies_detected,
                    'type': 'anomaly'
                }
                
            except Exception as e:
                print(f"Warning: Error training anomaly model {name}: {str(e)}")
        
        # 5. Time Series Analysis (if applicable)
        if 'datetime' in df.columns:
            self._train_time_series_models(df)
        
        self.models = results
        return results
    
    def _get_regression_models(self):
        """Get regression models based on available libraries"""
        if ADVANCED_LIBS:
            return {
                'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
                'LightGBM': LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
                'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42)
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
            }
    
    def _get_classification_models(self):
        """Get classification models based on available libraries"""
        if ADVANCED_LIBS:
            return {
                'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
                'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            }
        else:
            return {
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
            }
    
    def _get_anomaly_models(self):
        """Get anomaly detection models"""
        return {
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
            'One-Class SVM': OneClassSVM(gamma='scale', nu=0.1),
            'Local Outlier Factor': LocalOutlierFactor(contamination=0.1),
            'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42)
        }
    
    def _train_time_series_models(self, df):
        """Train time series models for outbreak detection"""
        ts_data = df.set_index('datetime').sort_index()
        
        outbreak_features = ['total_pathogen_detections', 'pathogen_risk_score', 'total_coliforms']
        
        for feature in outbreak_features:
            if feature in ts_data.columns:
                # Calculate rolling statistics
                ts_data[f'{feature}_ma_24h'] = ts_data[feature].rolling(window=24).mean()
                ts_data[f'{feature}_std_24h'] = ts_data[feature].rolling(window=24).std()
                
                # Z-score for anomaly detection
                ts_data[f'{feature}_zscore'] = ((ts_data[feature] - ts_data[f'{feature}_ma_24h']) / 
                                              (ts_data[f'{feature}_std_24h'] + 1e-6))
                
                # Outbreak threshold (Z-score > 2.5)
                ts_data[f'{feature}_outbreak_flag'] = ts_data[f'{feature}_zscore'].abs() > 2.5
        
        self.time_series_models['outbreak_detection'] = ts_data
    
    def predict_contamination_risk(self, input_data):
        """Make predictions for contamination risk"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        # Prepare features
        X_input = self.prepare_features(input_data)
        X_features = X_input[self.feature_columns]
        X_features = X_features.fillna(X_features.median())
        
        # Scale features
        X_scaled = self.scalers['pathogen'].transform(X_features)
        
        predictions = {}
        
        # Pathogen risk prediction
        if 'pathogen_best' in self.models:
            pathogen_pred = self.models['pathogen_best']['model'].predict(X_scaled)[0]
            predictions['pathogen_risk_score'] = max(0, pathogen_pred)
            predictions['pathogen_model'] = self.models['pathogen_best']['name']
        
        # Mycotoxin risk prediction
        if 'mycotoxin_best' in self.models:
            mycotoxin_pred = self.models['mycotoxin_best']['model'].predict(X_scaled)[0]
            predictions['mycotoxin_risk_score'] = max(0, mycotoxin_pred)
            predictions['mycotoxin_model'] = self.models['mycotoxin_best']['name']
        
        # Overall risk classification
        if 'classification_best' in self.models:
            risk_category = self.models['classification_best']['model'].predict(X_scaled)[0]
            risk_proba = self.models['classification_best']['model'].predict_proba(X_scaled)[0]
            
            predictions['risk_category'] = risk_category
            predictions['risk_probabilities'] = dict(zip(
                self.models['classification_best']['model'].classes_, risk_proba
            ))
            predictions['classification_model'] = self.models['classification_best']['name']
        
        # Anomaly detection
        anomaly_results = {}
        for model_name, model_info in self.models.items():
            if 'anomaly_' in model_name and model_info['type'] == 'anomaly':
                model = model_info['model']
                if hasattr(model, 'predict'):
                    try:
                        anomaly_score = model.predict(X_scaled)[0]
                        is_anomaly = anomaly_score == -1
                        anomaly_results[model_name.replace('anomaly_', '')] = {
                            'is_anomaly': is_anomaly,
                            'score': anomaly_score
                        }
                    except:
                        pass
        
        predictions['anomaly_detection'] = anomaly_results
        
        # Calculate overall contamination risk
        pathogen_risk = predictions.get('pathogen_risk_score', 0)
        mycotoxin_risk = predictions.get('mycotoxin_risk_score', 0)
        
        # Environmental factors from input
        environmental_risk = 0
        if not input_data.empty:
            sample = input_data.iloc[0]
            environmental_risk = (
                max(0, (sample.get('turbidity', 5) - 5) * 2) +
                max(0, (sample.get('bod_5', 20) - 20) * 0.5) +
                max(0, abs(sample.get('ph_level', 7) - 7) * 10) +
                max(0, (8 - sample.get('dissolved_oxygen', 8)) * 5)
            )
        
        total_risk = pathogen_risk + mycotoxin_risk + environmental_risk
        predictions['total_contamination_risk'] = total_risk
        
        return predictions