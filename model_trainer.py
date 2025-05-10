import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TORCSModelTrainer:
    def __init__(self):
        # Only include tracks that have data for now
        self.tracks = ['road', 'oval', 'dirt']  # Removed 'dirt' until data is available
        self.models = {
            track: {
                'steer': None,
                'accel': None,
                'brake': None
            } for track in self.tracks
        }
        self.scalers = {
            track: {
                'steer': StandardScaler(),
                'accel': StandardScaler(),
                'brake': StandardScaler()
            } for track in self.tracks
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
            print(f"\nHeaders in {csv_path}:")
            print(df.columns.tolist())
            return df.columns.tolist()
        except Exception as e:
            print(f"Error reading CSV headers from {csv_path}: {str(e)}")
            return None
    
    def prepare_features(self, df):
        """Prepare feature set from raw data"""
        # Define only essential features
        all_features = [
            'Angle',              # Car's angle relative to track
            'CurrentLapTime',     # Current lap time
            'DistanceFromStart',  # Distance from start line
            'DistanceCovered',    # Total distance covered
            'SpeedX',            # Longitudinal speed
            'SpeedY',            # Lateral speed
            'TrackPosition',      # Position relative to track center
            'RPM'                # Engine RPM
        ]
        
        # Add track sensors (these are crucial for navigation)
        all_features.extend([f'Track_{i}' for i in range(1, 20)])
        
        # Check which features are actually present in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        # Convert all columns to float
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna(subset=available_features)
        
        return df[available_features]
    
    def prepare_targets(self, df):
        """Prepare target variables"""
        # Map of internal names to CSV column names
        target_columns = {
            'steer': 'Steering',
            'accel': 'Acceleration',
            'brake': 'Braking'
        }
        
        # Convert target columns to float
        for internal_name, csv_name in target_columns.items():
            df[csv_name] = pd.to_numeric(df[csv_name], errors='coerce')
        
        # Drop any rows with NaN values in targets
        df = df.dropna(subset=list(target_columns.values()))
        
        return {
            'steer': df['Steering'],
            'accel': df['Acceleration'],
            'brake': df['Braking']
        }
    
    def train_track_model(self, track_name, csv_path):
        """Train models for a specific track"""
        try:
            print(f"\nTraining models for {track_name} track...")
            print(f"Loading data from {csv_path}...")
            
            # Check CSV headers first
            headers = self.check_csv_headers(csv_path)
            if headers is None:
                return
            
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"Loaded {len(df)} rows of data")
            
            # Prepare features and targets
            print("Preparing features and targets...")
            X = self.prepare_features(df)
            y = self.prepare_targets(df)
            
            print(f"Features shape: {X.shape}")
            print(f"Targets shape: {y['steer'].shape}")
            
            # Train models for each control action
            for action in ['steer', 'accel', 'brake']:
                print(f"\nTraining {action} model for {track_name}...")
                
                # Split data for this action
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y[action], test_size=0.2, random_state=42
                )
                
                print(f"Training set size: {len(X_train)}")
                print(f"Test set size: {len(X_test)}")
                
                # Scale features
                X_scaled = self.scalers[track_name][action].fit_transform(X_train)
                X_test_scaled = self.scalers[track_name][action].transform(X_test)
                
                # Initialize and train model with optimized parameters
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=500,  # Increased from 200
                    learning_rate=0.1,  # Increased from 0.05
                    max_depth=8,        # Increased from 6
                    min_child_weight=1, # Decreased from 2
                    subsample=0.9,      # Increased from 0.8
                    colsample_bytree=0.9, # Increased from 0.8
                    gamma=0.05,         # Decreased from 0.1
                    reg_alpha=0.05,     # Decreased from 0.1
                    reg_lambda=0.5,     # Decreased from 1
                    random_state=42
                )
                
                print(f"Fitting {action} model...")
                model.fit(X_scaled, y_train)
                
                self.models[track_name][action] = model
                
                # Create track directory if it doesn't exist
                track_dir = f'models/{track_name}'
                if not os.path.exists(track_dir):
                    os.makedirs(track_dir)
                
                # Save model and scaler
                print(f"Saving {action} model and scaler to {track_dir}...")
                model_path = f'{track_dir}/{action}_model.joblib'
                scaler_path = f'{track_dir}/{action}_scaler.joblib'
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[track_name][action], scaler_path)
                
                print(f"Model saved to {model_path}")
                print(f"Scaler saved to {scaler_path}")
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                print(f"{track_name} {action} model RMSE: {rmse:.4f}")
                
                # Print feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                print(f"\nTop 10 important features for {track_name} {action}:")
                print(feature_importance.head(10))
                
                # Print prediction statistics
                print(f"\nPrediction statistics for {action}:")
                print(f"Min prediction: {np.min(y_pred):.4f}")
                print(f"Max prediction: {np.max(y_pred):.4f}")
                print(f"Mean prediction: {np.mean(y_pred):.4f}")
                print(f"Std prediction: {np.std(y_pred):.4f}")
                
        except FileNotFoundError:
            print(f"Error: Could not find data file at {csv_path}")
            print(f"Please ensure the {track_name} race data CSV file exists in the logs directory.")
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
        """Load trained models and scalers for specified track or all tracks"""
        tracks_to_load = [track_name] if track_name else self.tracks
        
        for track in tracks_to_load:
            print(f"\nLoading models for {track} track...")
            for action in ['steer', 'accel', 'brake']:
                model_path = f'models/{track}/{action}_model.joblib'
                scaler_path = f'models/{track}/{action}_scaler.joblib'
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        self.models[track][action] = joblib.load(model_path)
                        self.scalers[track][action] = joblib.load(scaler_path)
                        print(f"Successfully loaded {track} {action} model and scaler")
                    except Exception as e:
                        print(f"Error loading {track} {action} model: {str(e)}")
                        self.models[track][action] = None
                        self.scalers[track][action] = None
                else:
                    print(f"Warning: {track} {action} model or scaler not found at:")
                    print(f"  Model: {model_path}")
                    print(f"  Scaler: {scaler_path}")
                    self.models[track][action] = None
                    self.scalers[track][action] = None

if __name__ == '__main__':
    trainer = TORCSModelTrainer()
    trainer.train_all_models() 