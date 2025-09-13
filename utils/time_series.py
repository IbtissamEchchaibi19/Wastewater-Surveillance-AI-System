"""
Time Series Analysis Module for Wastewater Surveillance
Implements time series forecasting and outbreak detection algorithms
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced time series libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    ADVANCED_MODELS = True
except ImportError:
    ADVANCED_MODELS = False

class TimeSeriesAnalyzer:
    """Advanced time series analysis for wastewater surveillance data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        self.outbreak_thresholds = {}
        self.seasonal_patterns = {}
        
    def prepare_time_series_data(self, df, target_column, datetime_column='datetime'):
        """Prepare time series data with feature engineering"""
        if datetime_column not in df.columns:
            raise ValueError(f"Datetime column '{datetime_column}' not found")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Sort by datetime
        ts_data = df.sort_values(datetime_column).copy()
        
        # Create time-based features
        ts_data['hour'] = ts_data[datetime_column].dt.hour
        ts_data['day_of_week'] = ts_data[datetime_column].dt.dayofweek
        ts_data['day_of_year'] = ts_data[datetime_column].dt.dayofyear
        ts_data['month'] = ts_data[datetime_column].dt.month
        ts_data['quarter'] = ts_data[datetime_column].dt.quarter
        ts_data['is_weekend'] = ts_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for periodic features
        ts_data['hour_sin'] = np.sin(2 * np.pi * ts_data['hour'] / 24)
        ts_data['hour_cos'] = np.cos(2 * np.pi * ts_data['hour'] / 24)
        ts_data['dow_sin'] = np.sin(2 * np.pi * ts_data['day_of_week'] / 7)
        ts_data['dow_cos'] = np.cos(2 * np.pi * ts_data['day_of_week'] / 7)
        ts_data['month_sin'] = np.sin(2 * np.pi * ts_data['month'] / 12)
        ts_data['month_cos'] = np.cos(2 * np.pi * ts_data['month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            ts_data[f'{target_column}_lag_{lag}'] = ts_data[target_column].shift(lag)
        
        # Rolling window features
        for window in [6, 12, 24, 48]:
            ts_data[f'{target_column}_mean_{window}h'] = ts_data[target_column].rolling(
                window=window, min_periods=1
            ).mean()
            ts_data[f'{target_column}_std_{window}h'] = ts_data[target_column].rolling(
                window=window, min_periods=1
            ).std()
            ts_data[f'{target_column}_min_{window}h'] = ts_data[target_column].rolling(
                window=window, min_periods=1
            ).min()
            ts_data[f'{target_column}_max_{window}h'] = ts_data[target_column].rolling(
                window=window, min_periods=1
            ).max()
        
        # Rate of change features
        ts_data[f'{target_column}_diff_1h'] = ts_data[target_column].diff(1)
        ts_data[f'{target_column}_diff_6h'] = ts_data[target_column].diff(6)
        ts_data[f'{target_column}_pct_change_1h'] = ts_data[target_column].pct_change(1)
        
        # Remove rows with NaN values (due to lags and rolling windows)
        ts_data = ts_data.dropna()
        
        return ts_data
    
    def detect_seasonal_patterns(self, df, target_column, datetime_column='datetime'):
        """Detect seasonal patterns in the time series"""
        ts_data = df.sort_values(datetime_column).copy()
        
        # Extract time components
        ts_data['hour'] = ts_data[datetime_column].dt.hour
        ts_data['day_of_week'] = ts_data[datetime_column].dt.dayofweek
        ts_data['month'] = ts_data[datetime_column].dt.month
        
        seasonal_patterns = {}
        
        # Hourly patterns
        hourly_mean = ts_data.groupby('hour')[target_column].agg(['mean', 'std'])
        seasonal_patterns['hourly'] = {
            'pattern': hourly_mean['mean'].to_dict(),
            'variability': hourly_mean['std'].to_dict(),
            'peak_hours': hourly_mean['mean'].nlargest(3).index.tolist(),
            'low_hours': hourly_mean['mean'].nsmallest(3).index.tolist()
        }
        
        # Daily patterns
        daily_mean = ts_data.groupby('day_of_week')[target_column].agg(['mean', 'std'])
        seasonal_patterns['daily'] = {
            'pattern': daily_mean['mean'].to_dict(),
            'variability': daily_mean['std'].to_dict(),
            'peak_days': daily_mean['mean'].nlargest(2).index.tolist(),
            'low_days': daily_mean['mean'].nsmallest(2).index.tolist()
        }
        
        # Monthly patterns
        monthly_mean = ts_data.groupby('month')[target_column].agg(['mean', 'std'])
        seasonal_patterns['monthly'] = {
            'pattern': monthly_mean['mean'].to_dict(),
            'variability': monthly_mean['std'].to_dict(),
            'peak_months': monthly_mean['mean'].nlargest(3).index.tolist(),
            'low_months': monthly_mean['mean'].nsmallest(3).index.tolist()
        }
        
        self.seasonal_patterns[target_column] = seasonal_patterns
        return seasonal_patterns
    
    def establish_baseline(self, df, target_column, datetime_column='datetime', baseline_days=30):
        """Establish baseline statistics for normal operation"""
        # Use first N days as baseline period
        df_sorted = df.sort_values(datetime_column)
        baseline_end = df_sorted[datetime_column].iloc[0] + timedelta(days=baseline_days)
        baseline_data = df_sorted[df_sorted[datetime_column] <= baseline_end]
        
        if len(baseline_data) == 0:
            raise ValueError("No data available for baseline period")
        
        baseline_stats = {
            'mean': baseline_data[target_column].mean(),
            'std': baseline_data[target_column].std(),
            'median': baseline_data[target_column].median(),
            'q25': baseline_data[target_column].quantile(0.25),
            'q75': baseline_data[target_column].quantile(0.75),
            'min': baseline_data[target_column].min(),
            'max': baseline_data[target_column].max(),
            'iqr': baseline_data[target_column].quantile(0.75) - baseline_data[target_column].quantile(0.25)
        }
        
        # Calculate outbreak thresholds
        self.outbreak_thresholds[target_column] = {
            'mild_threshold': baseline_stats['mean'] + 2 * baseline_stats['std'],
            'moderate_threshold': baseline_stats['mean'] + 2.5 * baseline_stats['std'],
            'severe_threshold': baseline_stats['mean'] + 3 * baseline_stats['std'],
            'extreme_threshold': baseline_stats['q75'] + 1.5 * baseline_stats['iqr']  # Outlier threshold
        }
        
        self.baseline_stats[target_column] = baseline_stats
        return baseline_stats
    
    def detect_outbreaks(self, df, target_column, datetime_column='datetime', method='statistical'):
        """Detect outbreak events using various methods"""
        if target_column not in self.baseline_stats:
            self.establish_baseline(df, target_column, datetime_column)
        
        ts_data = df.sort_values(datetime_column).copy()
        baseline = self.baseline_stats[target_column]
        thresholds = self.outbreak_thresholds[target_column]
        
        outbreak_results = []
        
        if method == 'statistical':
            # Z-score based detection
            ts_data['zscore'] = zscore(ts_data[target_column])
            
            # Multiple threshold levels
            ts_data['mild_outbreak'] = ts_data[target_column] > thresholds['mild_threshold']
            ts_data['moderate_outbreak'] = ts_data[target_column] > thresholds['moderate_threshold']
            ts_data['severe_outbreak'] = ts_data[target_column] > thresholds['severe_threshold']
            ts_data['extreme_outbreak'] = ts_data[target_column] > thresholds['extreme_threshold']
            
            # Overall outbreak flag
            ts_data['outbreak'] = (ts_data['mild_outbreak'] | 
                                 (np.abs(ts_data['zscore']) > 2.5))
            
        elif method == 'cusum':
            # CUSUM (Cumulative Sum) control chart
            target_mean = baseline['mean']
            target_std = baseline['std']
            
            # Standardize the data
            standardized = (ts_data[target_column] - target_mean) / target_std
            
            # CUSUM parameters
            k = 0.5  # Reference value
            h = 4    # Decision interval
            
            # Calculate CUSUM
            cusum_pos = np.zeros(len(standardized))
            cusum_neg = np.zeros(len(standardized))
            
            for i in range(1, len(standardized)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i] - k)
                cusum_neg[i] = max(0, cusum_neg[i-1] - standardized.iloc[i] - k)
            
            # Detect outbreaks when CUSUM exceeds threshold
            ts_data['cusum_pos'] = cusum_pos
            ts_data['cusum_neg'] = cusum_neg
            ts_data['outbreak'] = (cusum_pos > h) | (cusum_neg > h)
            
        elif method == 'ewma':
            # Exponentially Weighted Moving Average
            target_mean = baseline['mean']
            target_std = baseline['std']
            
            lambda_param = 0.2  # Smoothing parameter
            L = 2.814  # Control limit multiplier for lambda=0.2
            
            # Calculate EWMA
            ewma = ts_data[target_column].ewm(alpha=lambda_param).mean()
            ewma_std = target_std * np.sqrt(lambda_param / (2 - lambda_param))
            
            # Control limits
            ucl = target_mean + L * ewma_std
            lcl = target_mean - L * ewma_std
            
            ts_data['ewma'] = ewma
            ts_data['outbreak'] = (ewma > ucl) | (ewma < lcl)
        
        # Compile outbreak events
        outbreak_periods = []
        if 'outbreak' in ts_data.columns:
            outbreak_data = ts_data[ts_data['outbreak'] == True]
            
            if not outbreak_data.empty:
                # Group consecutive outbreak points
                outbreak_data['group'] = (
                    outbreak_data[datetime_column].diff() > pd.Timedelta(hours=2)
                ).cumsum()
                
                for group_id, group in outbreak_data.groupby('group'):
                    outbreak_periods.append({
                        'start_time': group[datetime_column].min(),
                        'end_time': group[datetime_column].max(),
                        'duration_hours': (group[datetime_column].max() - group[datetime_column].min()).total_seconds() / 3600,
                        'peak_value': group[target_column].max(),
                        'mean_value': group[target_column].mean(),
                        'severity': self._classify_outbreak_severity(group[target_column].max(), thresholds)
                    })
        
        return {
            'outbreak_periods': outbreak_periods,
            'total_outbreaks': len(outbreak_periods),
            'time_series_data': ts_data,
            'method_used': method
        }
    
    def _classify_outbreak_severity(self, peak_value, thresholds):
        """Classify outbreak severity based on peak value"""
        if peak_value >= thresholds['extreme_threshold']:
            return 'extreme'
        elif peak_value >= thresholds['severe_threshold']:
            return 'severe'
        elif peak_value >= thresholds['moderate_threshold']:
            return 'moderate'
        elif peak_value >= thresholds['mild_threshold']:
            return 'mild'
        else:
            return 'normal'
    
    def train_forecasting_models(self, df, target_column, datetime_column='datetime', forecast_horizon=24):
        """Train time series forecasting models"""
        # Prepare time series features
        ts_data = self.prepare_time_series_data(df, target_column, datetime_column)
        
        # Define features for modeling
        feature_cols = [col for col in ts_data.columns if col not in [datetime_column, target_column]]
        feature_cols = [col for col in feature_cols if not ts_data[col].isnull().all()]
        
        X = ts_data[feature_cols]
        y = ts_data[target_column]
        
        # Split data (use last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        
        # Initialize models
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        }
        
        if ADVANCED_MODELS:
            models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        results = {}
        
        # Train and evaluate models
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Calculate percentage error
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'predictions': y_pred,
                    'actual': y_test.values
                }
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        # Select best model based on RMSE
        if results:
            best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
            results['best'] = results[best_model_name].copy()
            results['best']['best_model_name'] = best_model_name
        
        self.models[target_column] = results
        return results
    
    def forecast(self, input_data, target_column, steps_ahead=24):
        """Generate forecasts using trained models"""
        if target_column not in self.models:
            raise ValueError(f"No trained model found for {target_column}")
        
        if 'best' not in self.models[target_column]:
            raise ValueError(f"No best model selected for {target_column}")
        
        model_info = self.models[target_column]['best']
        model = model_info['model']
        scaler = self.scalers[target_column]
        
        # Prepare features for the last known data point
        ts_data = self.prepare_time_series_data(input_data, target_column)
        
        if ts_data.empty:
            raise ValueError("No valid data for forecasting")
        
        # Get features (excluding datetime and target)
        feature_cols = [col for col in ts_data.columns if col not in ['datetime', target_column]]
        feature_cols = [col for col in feature_cols if col in scaler.feature_names_in_]
        
        forecasts = []
        current_data = ts_data.iloc[-1:].copy()  # Start with last known point
        
        for step in range(steps_ahead):
            # Prepare features for current step
            X_current = current_data[feature_cols]
            X_scaled = scaler.transform(X_current)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            forecasts.append(prediction)
            
            # Update data for next step (simplified - in practice, need more sophisticated updating)
            # Update lag features with the prediction
            for lag in [1, 2, 3, 6, 12, 24]:
                lag_col = f'{target_column}_lag_{lag}'
                if lag_col in current_data.columns:
                    if lag == 1:
                        current_data[lag_col] = prediction
                    else:
                        # Shift existing lag values
                        prev_lag_col = f'{target_column}_lag_{lag-1}'
                        if prev_lag_col in current_data.columns:
                            current_data[lag_col] = current_data[prev_lag_col]
        
        return forecasts
    
    def detect_anomalies_in_time_series(self, df, target_column, datetime_column='datetime', 
                                      method='isolation_forest', contamination=0.1):
        """Detect anomalies in time series data"""
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import EllipticEnvelope
        
        ts_data = self.prepare_time_series_data(df, target_column, datetime_column)
        
        # Select features for anomaly detection
        feature_cols = [col for col in ts_data.columns 
                       if col not in [datetime_column, target_column] and 
                       not ts_data[col].isnull().all()]
        
        X = ts_data[feature_cols].fillna(ts_data[feature_cols].median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        anomaly_labels = detector.fit_predict(X_scaled)
        anomaly_scores = detector.decision_function(X_scaled) if hasattr(detector, 'decision_function') else anomaly_labels
        
        # Add results to time series data
        ts_data['anomaly'] = anomaly_labels == -1
        ts_data['anomaly_score'] = anomaly_scores
        
        # Identify anomaly periods
        anomaly_periods = []
        if ts_data['anomaly'].any():
            anomaly_data = ts_data[ts_data['anomaly'] == True]
            
            # Group consecutive anomalies
            anomaly_data['group'] = (
                anomaly_data[datetime_column].diff() > pd.Timedelta(hours=2)
            ).cumsum()
            
            for group_id, group in anomaly_data.groupby('group'):
                anomaly_periods.append({
                    'start_time': group[datetime_column].min(),
                    'end_time': group[datetime_column].max(),
                    'duration_hours': (group[datetime_column].max() - group[datetime_column].min()).total_seconds() / 3600,
                    'peak_score': group['anomaly_score'].min(),  # More negative = more anomalous for isolation forest
                    'mean_value': group[target_column].mean(),
                    'severity': 'high' if group['anomaly_score'].min() < -0.5 else 'medium'
                })
        
        return {
            'anomaly_periods': anomaly_periods,
            'total_anomalies': len(anomaly_periods),
            'time_series_data': ts_data,
            'method_used': method
        }
    
    def calculate_trend_analysis(self, df, target_column, datetime_column='datetime', window_days=7):
        """Calculate trend analysis for the time series"""
        ts_data = df.sort_values(datetime_column).copy()
        
        # Calculate rolling trends
        window_hours = window_days * 24
        ts_data['rolling_mean'] = ts_data[target_column].rolling(
            window=window_hours, min_periods=1
        ).mean()
        
        # Calculate trend using linear regression over rolling windows
        trends = []
        for i in range(window_hours, len(ts_data)):
            window_data = ts_data.iloc[i-window_hours:i]
            X = np.arange(len(window_data)).reshape(-1, 1)
            y = window_data[target_column].values
            
            if len(np.unique(y)) > 1:  # Avoid constant values
                lr = LinearRegression()
                lr.fit(X, y)
                trend_slope = lr.coef_[0]
            else:
                trend_slope = 0
            
            trends.append(trend_slope)
        
        # Pad trends with zeros for first window_hours points
        trends = [0] * window_hours + trends
        ts_data['trend_slope'] = trends
        
        # Classify trend direction
        def classify_trend(slope):
            if slope > 0.5:
                return 'strongly_increasing'
            elif slope > 0.1:
                return 'increasing'
            elif slope > -0.1:
                return 'stable'
            elif slope > -0.5:
                return 'decreasing'
            else:
                return 'strongly_decreasing'
        
        ts_data['trend_direction'] = ts_data['trend_slope'].apply(classify_trend)
        
        # Calculate overall trend for recent period
        recent_data = ts_data.tail(window_hours)
        if len(recent_data) > 1:
            X_recent = np.arange(len(recent_data)).reshape(-1, 1)
            y_recent = recent_data[target_column].values
            
            lr_recent = LinearRegression()
            lr_recent.fit(X_recent, y_recent)
            overall_trend_slope = lr_recent.coef_[0]
            overall_trend_direction = classify_trend(overall_trend_slope)
        else:
            overall_trend_slope = 0
            overall_trend_direction = 'stable'
        
        return {
            'time_series_data': ts_data,
            'overall_trend_slope': overall_trend_slope,
            'overall_trend_direction': overall_trend_direction,
            'trend_analysis_window_days': window_days
        }
    
    def generate_time_series_report(self, df, target_column, datetime_column='datetime'):
        """Generate comprehensive time series analysis report"""
        try:
            # Establish baseline
            baseline = self.establish_baseline(df, target_column, datetime_column)
            
            # Detect seasonal patterns
            seasonal = self.detect_seasonal_patterns(df, target_column, datetime_column)
            
            # Detect outbreaks
            outbreak_analysis = self.detect_outbreaks(df, target_column, datetime_column, method='statistical')
            
            # Trend analysis
            trend_analysis = self.calculate_trend_analysis(df, target_column, datetime_column)
            
            # Anomaly detection
            anomaly_analysis = self.detect_anomalies_in_time_series(df, target_column, datetime_column)
            
            # Summary statistics
            recent_data = df.sort_values(datetime_column).tail(24)  # Last 24 hours
            current_stats = {
                'current_mean': recent_data[target_column].mean(),
                'current_std': recent_data[target_column].std(),
                'current_max': recent_data[target_column].max(),
                'current_min': recent_data[target_column].min(),
                'deviation_from_baseline': (recent_data[target_column].mean() - baseline['mean']) / baseline['std']
            }
            
            # Risk assessment
            risk_level = 'low'
            if current_stats['deviation_from_baseline'] > 3:
                risk_level = 'critical'
            elif current_stats['deviation_from_baseline'] > 2:
                risk_level = 'high'
            elif current_stats['deviation_from_baseline'] > 1:
                risk_level = 'medium'
            
            report = {
                'target_column': target_column,
                'analysis_period': {
                    'start': df[datetime_column].min(),
                    'end': df[datetime_column].max(),
                    'total_hours': (df[datetime_column].max() - df[datetime_column].min()).total_seconds() / 3600
                },
                'baseline_stats': baseline,
                'seasonal_patterns': seasonal,
                'outbreak_analysis': outbreak_analysis,
                'trend_analysis': trend_analysis,
                'anomaly_analysis': anomaly_analysis,
                'current_stats': current_stats,
                'risk_level': risk_level
            }
            
            return report
            
        except Exception as e:
            return {
                'error': f"Failed to generate time series report: {str(e)}",
                'target_column': target_column
            }