import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TORCSModelTrainer:
    def __init__(self):
        self.models = {
            'steer': None,
            'accel': None,
            'brake': None
        }
        self.scalers = {
            'steer': StandardScaler(),
            'accel': StandardScaler(),
            'brake': StandardScaler()
        }
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def prepare_features(self, df):
        """Prepare feature set from raw data"""
        # Select relevant features for prediction
        feature_columns = [
            'speed_x', 'speed_y', 'speed_z', 'rpm', 'gear',
            'angle', 'track_position', 'track_edge_dist'
        ]
        
        # Add track sensors
        feature_columns.extend([f'track_sensor_{i}' for i in range(19)])
        
        # Add opponent sensors
        feature_columns.extend([f'opponent_sensor_{i}' for i in range(36)])
        
        # Convert all columns to float
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna(subset=feature_columns)
        
        return df[feature_columns]
    
    def prepare_targets(self, df):
        """Prepare target variables"""
        # Convert target columns to float
        for action in ['steer', 'accel', 'brake']:
            df[action] = pd.to_numeric(df[action], errors='coerce')
        
        # Drop any rows with NaN values in targets
        df = df.dropna(subset=['steer', 'accel', 'brake'])
        
        return {
            'steer': df['steer'],
            'accel': df['accel'],
            'brake': df['brake']
        }
    
    def train_models(self, csv_path='src/logs/combined_race_data.csv'):
        """Train XGBoost models for steering, acceleration, and braking"""
        try:
            # Load and prepare data
            print(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Prepare features and targets
            print("Preparing features and targets...")
            X = self.prepare_features(df)
            y = self.prepare_targets(df)
            
            # Train models for each control action
            for action in ['steer', 'accel', 'brake']:
                print(f"\nTraining {action} model...")
                
                # Split data for this action
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y[action], test_size=0.2, random_state=42
                )
                
                # Scale features
                X_scaled = self.scalers[action].fit_transform(X_train)
                X_test_scaled = self.scalers[action].transform(X_test)
                
                # Initialize and train model
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                
                print(f"Fitting {action} model...")
                model.fit(
                    X_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                self.models[action] = model
                
                # Save model and scaler
                print(f"Saving {action} model and scaler...")
                joblib.dump(model, f'models/{action}_model.joblib')
                joblib.dump(self.scalers[action], f'models/{action}_scaler.joblib')
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = np.mean((y_test - y_pred) ** 2)
                print(f"{action} model MSE: {mse:.4f}")
                
        except FileNotFoundError:
            print(f"Error: Could not find data file at {csv_path}")
            print("Please ensure the race data CSV file exists in the logs directory.")
        except Exception as e:
            print(f"Error during model training: {str(e)}")
    
    def load_models(self):
        """Load trained models and scalers"""
        for action in ['steer', 'accel', 'brake']:
            model_path = f'models/{action}_model.joblib'
            scaler_path = f'models/{action}_scaler.joblib'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[action] = joblib.load(model_path)
                    self.scalers[action] = joblib.load(scaler_path)
                    print(f"Successfully loaded {action} model and scaler")
                except Exception as e:
                    print(f"Error loading {action} model: {str(e)}")
                    self.models[action] = None
                    self.scalers[action] = None
            else:
                print(f"Warning: {action} model or scaler not found")
                self.models[action] = None
                self.scalers[action] = None

if __name__ == '__main__':
    trainer = TORCSModelTrainer()
    trainer.train_models() 