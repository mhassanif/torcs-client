import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os
import time
from typing import Dict, List, Tuple

class TORCSModelTrainer:
    def __init__(self):
        # Only include road track for now
        self.tracks = ['road']  # Commented out other tracks: 'oval', 'dirt'
        self.models = {
            track: {
                'steer': None,
                'accel': None,
                'brake': None,
                'clutch': None,
                'gear': None
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
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set from raw data"""
        try:
            # Define features to use (excluding targets and ignored features)
            feature_columns = [
                'Angle', 'CurrentLapTime', 'DistanceFromStart', 
                'DistanceCovered', 'RacePosition', 'RPM', 
                'SpeedX', 'SpeedY', 'SpeedZ', 'TrackPosition', 'Z'
            ]
            
            # Add opponent sensors
            feature_columns.extend([f'Opponent_{i}' for i in range(1, 37)])
            
            # Add track sensors
            feature_columns.extend([f'Track_{i}' for i in range(1, 20)])
            
            # Check which features are actually present
            available_features = [col for col in feature_columns if col in df.columns]
            if not available_features:
                raise ValueError("No features found in the dataset")
            
            print(f"Using {len(available_features)} features out of {len(feature_columns)} defined features")
            
            # Convert all columns to float and handle NaN values
            df = df[available_features].astype(float)
            
            # Check for and handle infinite values
            if np.isinf(df.values).any():
                print("Warning: Found infinite values, replacing with NaN")
                df = df.replace([np.inf, -np.inf], np.nan)
            
            # Handle NaN values
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                print(f"Warning: Found {nan_count} NaN values, filling with forward then backward fill")
                df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            raise
    
    def prepare_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare target variables"""
        try:
            # Map of internal names to CSV column names
            target_columns = {
                'steer': 'Steering',
                'accel': 'Acceleration',
                'brake': 'Braking',
                'clutch': 'Clutch',
                'gear': 'Gear'
            }
            
            # Convert target columns to float and handle NaN values
            targets = {}
            missing_targets = []
            
            for internal_name, csv_name in target_columns.items():
                if csv_name not in df.columns:
                    missing_targets.append(csv_name)
                    continue
                    
                targets[internal_name] = pd.to_numeric(df[csv_name], errors='coerce')
                
                # Check for and handle infinite values
                if np.isinf(targets[internal_name]).any():
                    print(f"Warning: Found infinite values in {csv_name}, replacing with NaN")
                    targets[internal_name] = targets[internal_name].replace([np.inf, -np.inf], np.nan)
                
                # Handle NaN values
                nan_count = targets[internal_name].isna().sum()
                if nan_count > 0:
                    print(f"Warning: Found {nan_count} NaN values in {csv_name}, filling with forward then backward fill")
                    targets[internal_name] = targets[internal_name].ffill().bfill()
                
                # Validate target ranges
                if internal_name == 'steer':
                    # Clip steering values to valid range
                    targets[internal_name] = targets[internal_name].clip(-1.0, 1.0)
                elif internal_name == 'accel':
                    # Convert acceleration to discrete values (0 or 1) during data preparation
                    targets[internal_name] = (targets[internal_name] > 0.1).astype(int)
                elif internal_name in ['brake', 'clutch']:
                    targets[internal_name] = targets[internal_name].clip(0.0, 1.0)
                elif internal_name == 'gear':
                    # Convert gear to integer and handle reverse gear (-1)
                    targets[internal_name] = targets[internal_name].round().astype(int)
                    # Ensure gear is between -1 and 6
                    targets[internal_name] = targets[internal_name].clip(-1, 6)
            
            if missing_targets:
                print(f"Warning: Missing target columns: {', '.join(missing_targets)}")
            
            if not targets:
                raise ValueError("No valid target columns found in the dataset")
            
            return targets
            
        except Exception as e:
            print(f"Error in prepare_targets: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimal preprocessing to the dataset"""
        print("\nPreprocessing data...")
        original_len = len(df)
        
        # Filter out redundant stationary rows
        print("Filtering out redundant stationary rows...")
        df['DistanceCovered_diff'] = df['DistanceCovered'].diff()
        race_starts = df[(df['DistanceCovered'] == 0) & (df['FuelLevel'] == 94)].index
        df = df[(df['DistanceCovered_diff'] > 0) | 
               (df.index.isin(race_starts)) | 
               (df['DistanceCovered_diff'].isna())]
        df = df.drop('DistanceCovered_diff', axis=1)
        print(f"After filtering: {len(df)} rows")
        
        return df
    
    def train_track_model(self, track_name: str, csv_path: str):
        """Train models for a specific track"""
        try:
            print(f"\nTraining models for {track_name} track...")
            
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # Load and prepare data
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                raise ValueError(f"Error reading CSV: {str(e)}")
            
            if len(df) == 0:
                raise ValueError("Empty dataset")
                
            print(f"Loaded {len(df)} rows of data")
            
            # Apply specific preprocessing
            df = self.preprocess_data(df)
            
            if len(df) == 0:
                raise ValueError("No data after preprocessing")
            
            # Filter out redundant stationary rows
            print("\nFiltering out redundant stationary rows...")
            df['DistanceCovered_diff'] = df['DistanceCovered'].diff()
            race_starts = df[(df['DistanceCovered'] == 0) & (df['FuelLevel'] == 94)].index
            df = df[(df['DistanceCovered_diff'] > 0) | 
                   (df.index.isin(race_starts)) | 
                   (df['DistanceCovered_diff'].isna())]
            df = df.drop('DistanceCovered_diff', axis=1)
            print(f"After filtering: {len(df)} rows")
            
            if len(df) == 0:
                raise ValueError("No data after filtering")
            
            # Prepare features and targets
            X = self.prepare_features(df)
            y = self.prepare_targets(df)
            
            print(f"\nFeature shape: {X.shape}")
            
            # Scale features using single scaler per track
            X_scaled = self.scalers[track_name].fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Define parameter grids for different actions
            regression_param_grid = {
                'hidden_layer_sizes': [(128,), (128, 64), (128, 64, 32)],
                'activation': ['relu'],
                'learning_rate_init': [0.001, 0.0001],
                'max_iter': [500],
                'batch_size': ['auto', 32, 64],
                'alpha': [0.0001, 0.001]
            }
            
            classification_param_grid = {
                'hidden_layer_sizes': [(128,), (128, 64), (128, 64, 32)],
                'activation': ['relu'],
                'learning_rate_init': [0.001, 0.0001],
                'max_iter': [500],
                'batch_size': ['auto', 32, 64],
                'alpha': [0.0001, 0.001]
            }
            
            # 2-fold cross-validation
            cv = KFold(n_splits=2, shuffle=True, random_state=42)
            
            # Train models for each control action
            for action in ['steer', 'accel', 'brake', 'clutch', 'gear']:
                if action not in y:
                    print(f"Skipping {action} - no target data")
                    continue
                    
                print(f"\nTraining {action} model...")
                start_time = time.time()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y[action], test_size=0.2, random_state=42
                )
                
                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"Skipping {action} - insufficient data")
                    continue
                
                print(f"Training set: {len(X_train)} rows")
                print(f"Test set: {len(X_test)} rows")
                
                # Initialize GridSearchCV with appropriate model type
                if action == 'gear':
                    base_model = MLPClassifier(random_state=42, early_stopping=True)
                    param_grid = classification_param_grid
                else:
                    base_model = MLPRegressor(random_state=42, early_stopping=True)
                    param_grid = regression_param_grid
                
                grid = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv,
                    verbose=1,
                    n_jobs=-1
                )
                
                # Train model
                try:
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    print(f"\nBest parameters for {action}:")
                    print(grid.best_params_)
                except Exception as e:
                    print(f"Training failed: {str(e)}")
                    continue
                
                self.models[track_name][action] = best_model
                
                # Save model and scaler
                try:
                    model_path = f'models/{track_name}/{action}_model.joblib'
                    joblib.dump(best_model, model_path)
                    
                    if action == 'steer':
                        scaler_path = f'models/{track_name}/scaler.joblib'
                        joblib.dump(self.scalers[track_name], scaler_path)
                except Exception as e:
                    print(f"Model save failed: {str(e)}")
                    continue
                
                # Evaluate model
                try:
                    y_pred = best_model.predict(X_test)
                    if action == 'gear':
                        # For gear classification, calculate accuracy
                        accuracy = np.mean(y_test == y_pred)
                        print(f"\nModel evaluation for {action}:")
                        print(f"Accuracy: {accuracy:.4f}")
                        print(f"Unique predicted gears: {np.unique(y_pred)}")
                    else:
                        # For regression tasks, calculate MSE and MAE
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(y_test - y_pred))
                        print(f"\nModel evaluation for {action}:")
                        print(f"RMSE: {rmse:.4f}")
                        print(f"MAE: {mae:.4f}")
                        print(f"Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
                    
                    elapsed = time.time() - start_time
                    print(f"Training time: {elapsed:.2f} seconds")
                except Exception as e:
                    print(f"Evaluation failed: {str(e)}")
                
        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def train_all_models(self):
        """Train models for all available tracks"""
        track_data = {
            'road': 'logs/road_merged.csv',
            # 'oval': 'logs/oval_merged.csv',  # Commented out for now
            # 'dirt': 'logs/dirt_merged.csv'   # Commented out for now
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
        for action in ['steer', 'accel', 'brake', 'clutch', 'gear']:
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