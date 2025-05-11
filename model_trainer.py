import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple

class TORCSModelTrainer:
    def __init__(self):
        # Only include tracks that have data for now
        self.tracks = ['road', 'oval', 'dirt']
        self.models = {
            track: {
                'steer': None,
                'accel': None,
                'brake': None
            } for track in self.tracks
        }
        # Single scaler per track
        self.scalers = {
            track: StandardScaler() for track in self.tracks
        }
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Create track-specific model directories
        for track in self.tracks:
            track_dir = f'models/{track}'
            if not os.path.exists(track_dir):
                os.makedirs(track_dir)
    
    def check_csv_headers(self, csv_path):
        """Check the headers of a CSV file"""
        try:
            df = pd.read_csv(csv_path, nrows=0)  # Only read headers
            return df.columns.tolist()
        except Exception as e:
            print(f"Error reading CSV headers from {csv_path}: {str(e)}")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set from raw data"""
        # Define base features
        base_features = [
            'Angle',              # Car's angle relative to track
            'SpeedX',            # Longitudinal speed
            'SpeedY',            # Lateral speed
            'TrackPosition',      # Position relative to track center
            'RPM'                # Engine RPM
        ]
        
        # Add track sensors
        base_features.extend([f'Track_{i}' for i in range(1, 20)])
        
        # Check which features are actually present
        available_features = [col for col in base_features if col in df.columns]
        
        # Convert all columns to float and handle NaN values
        df = df[available_features].astype(float)
        df = df.ffill().bfill()
        
        return df
    
    def prepare_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare target variables"""
        # Map of internal names to CSV column names
        target_columns = {
            'steer': 'Steering',
            'accel': 'Acceleration',
            'brake': 'Braking'
        }
        
        # Convert target columns to float and handle NaN values
        targets = {}
        for internal_name, csv_name in target_columns.items():
            targets[internal_name] = pd.to_numeric(df[csv_name], errors='coerce')
            targets[internal_name] = targets[internal_name].ffill().bfill()
        
        return targets
    
    def train_track_model(self, track_name: str, csv_path: str):
        """Train models for a specific track with improved parameters"""
        try:
            print(f"\nTraining models for {track_name} track...")
            
            # Load and prepare data
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"Loaded {len(df)} rows of data")
            
            # Validate data
            print("\nValidating data...")
            print("Target value ranges:")
            for col in ['Steering', 'Acceleration', 'Braking']:
                if col in df.columns:
                    print(f"{col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")
                else:
                    print(f"Warning: {col} column not found in data")
            
            # Check for missing values
            missing = df[['Steering', 'Acceleration', 'Braking']].isnull().sum()
            print("\nMissing values in targets:")
            print(missing)
            
            # Prepare features and targets
            X = self.prepare_features(df)
            y = self.prepare_targets(df)
            
            print(f"\nFeature shape: {X.shape}")
            print("Feature value ranges:")
            for col in X.columns[:5]:  # Print first 5 features
                print(f"{col}: min={X[col].min():.3f}, max={X[col].max():.3f}, mean={X[col].mean():.3f}")
            
            # Scale features using single scaler per track
            X_scaled = self.scalers[track_name].fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Train models for each control action
            for action in ['steer', 'accel', 'brake']:
                print(f"\nTraining {action} model...")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y[action], test_size=0.2, random_state=42
                )
                
                print(f"Training set size: {len(X_train)}")
                print(f"Test set size: {len(X_test)}")
                print(f"Target distribution - min: {y_train.min():.3f}, max: {y_train.max():.3f}, mean: {y_train.mean():.3f}")
                
                # Initialize XGBoost model with improved parameters
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=2000,
                    learning_rate=0.01,
                    max_depth=8,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    early_stopping_rounds=100,
                    eval_metric=['rmse', 'mae']
                )
                
                # Train with early stopping
                eval_set = [(X_train, y_train), (X_test, y_test)]
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=True
                )
                
                self.models[track_name][action] = model
                
                # Save model and scaler
                model_path = f'models/{track_name}/{action}_model.joblib'
                joblib.dump(model, model_path)
                
                # Save scaler only once per track
                if action == 'steer':
                    scaler_path = f'models/{track_name}/scaler.joblib'
                    joblib.dump(self.scalers[track_name], scaler_path)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                
                print(f"\nFinal model evaluation for {action}:")
                print(f"RMSE: {rmse:.4f} (Target range: {y_test.min():.3f} to {y_test.max():.3f})")
                print(f"MAE: {mae:.4f}")
                print(f"Prediction range - min: {y_pred.min():.3f}, max: {y_pred.max():.3f}, mean: {y_pred.mean():.3f}")
                
                # Calculate and print error distribution
                errors = y_test - y_pred
                print(f"Error distribution:")
                print(f"  Mean error: {errors.mean():.4f}")
                print(f"  Std error: {errors.std():.4f}")
                print(f"  Max positive error: {errors.max():.4f}")
                print(f"  Max negative error: {errors.min():.4f}")
                
                # Print feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                print(f"\nTop 5 important features for {action}:")
                print(feature_importance.head(5))
                
        except Exception as e:
            print(f"Error during model training for {track_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def train_all_models(self):
        """Train models for all available tracks"""
        track_data = {
            'road': 'logs/road_merged.csv',
            'oval': 'logs/oval_merged.csv',
            'dirt': 'logs/dirt_merged.csv'
        }
        
        for track, csv_path in track_data.items():
            if os.path.exists(csv_path):
                self.train_track_model(track, csv_path)
            else:
                print(f"Warning: Data file for {track} track not found at {csv_path}")
    
    def load_models(self, track_name=None):
        """Load trained models and scaler for specified track"""
        if track_name is None:
            print("Error: No track type specified for model loading")
            return
            
        print(f"\nLoading models for {track_name} track...")
        
        # Load scaler first
        scaler_path = f'models/{track_name}/scaler.joblib'
        if os.path.exists(scaler_path):
            try:
                self.scalers[track_name] = joblib.load(scaler_path)
                print(f"Successfully loaded {track_name} scaler")
            except Exception as e:
                print(f"Error loading {track_name} scaler: {str(e)}")
                self.scalers[track_name] = StandardScaler()
        else:
            print(f"Warning: {track_name} scaler not found at {scaler_path}")
            self.scalers[track_name] = StandardScaler()
        
        # Load models
        for action in ['steer', 'accel', 'brake']:
            model_path = f'models/{track_name}/{action}_model.joblib'
            if os.path.exists(model_path):
                try:
                    self.models[track_name][action] = joblib.load(model_path)
                    print(f"Successfully loaded {track_name} {action} model")
                except Exception as e:
                    print(f"Error loading {track_name} {action} model: {str(e)}")
                    self.models[track_name][action] = None
            else:
                print(f"Warning: {track_name} {action} model not found at {model_path}")
                self.models[track_name][action] = None

if __name__ == '__main__':
    trainer = TORCSModelTrainer()
    trainer.train_all_models() 