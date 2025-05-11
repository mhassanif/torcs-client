import msgParser
import carState
import carControl
import time
import numpy as np
from data_logger import DataLogger
from model_trainer import TORCSModelTrainer
import pandas as pd

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage, car_type='unknown'):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.car_type = car_type
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Initialize model trainer
        self.model_trainer = TORCSModelTrainer()
        
        # Track type mapping (1=road, 2=oval, 3=dirt)
        self.track_type_map = {
            1: 'road',
            2: 'oval',
            3: 'dirt'
        }
        
        # Current track type
        self.current_track_type = None
        
        print(f"Debug - Driver initialized with stage: {stage}")
        print(f"Debug - Track mapping: {self.track_type_map}")
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def prepare_features_for_prediction(self):
        """Prepare features for model prediction"""
        # Get current state features
        current_features = {}
        
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
        
        # Add base features
        current_features['Angle'] = self.state.getAngle()
        current_features['CurrentLapTime'] = self.state.getCurLapTime()
        current_features['DistanceFromStart'] = self.state.getDistFromStart()
        current_features['DistanceCovered'] = self.state.getDistRaced()
        current_features['RacePosition'] = self.state.getRacePos()
        current_features['RPM'] = self.state.getRpm()
        current_features['SpeedX'] = self.state.getSpeedX()
        current_features['SpeedY'] = self.state.getSpeedY()
        current_features['SpeedZ'] = self.state.getSpeedZ()
        current_features['TrackPosition'] = self.state.getTrackPos()
        current_features['Z'] = self.state.getZ()
        
        # Add opponent sensors
        opponents = self.state.getOpponents()
        if opponents is not None:
            for i, value in enumerate(opponents, 1):
                current_features[f'Opponent_{i}'] = value
        
        # Add track sensors
        track_sensors = self.state.getTrack()
        if track_sensors is not None:
            for i, value in enumerate(track_sensors, 1):
                current_features[f'Track_{i}'] = value
        
        # Convert to DataFrame
        current_df = pd.DataFrame([current_features])
        
        # Check which features are actually present
        available_features = [col for col in feature_columns if col in current_df.columns]
        
        # Convert all columns to float and handle NaN values
        current_df = current_df[available_features].astype(float)
        current_df = current_df.ffill().bfill()
        
        # Get feature names from the model (using steer model as reference)
        if self.current_track_type is None:
            track_type = 'road'  # Default to road track
        else:
            track_type = self.current_track_type
            
        if self.model_trainer.models[track_type]['steer'] is not None:
            expected_features = self.model_trainer.models[track_type]['steer'].feature_names_in_
            # Ensure columns are in the same order as expected by the model
            current_df = current_df[expected_features]
        
        return current_df.values
    
    def predict_controls(self, features):
        """Predict control actions using track-specific MLPRegressor models"""
        predictions = {}
        
        if self.current_track_type is None:
            print("Warning: No track type detected, using default models")
            track_type = 'road'  # Default to road track
        else:
            track_type = self.current_track_type
            print(f"Using models for {track_type} track")
        
        print(f"\nDebug - Features shape before scaling: {features.shape}")
        
        for action in ['steer', 'accel', 'brake', 'clutch', 'gear']:
            if self.model_trainer.models[track_type][action] is not None:
                try:
                    # Get feature names from the model
                    feature_names = self.model_trainer.models[track_type][action].feature_names_in_
                    print(f"\nDebug - Model expects {len(feature_names)} features")
                    print(f"Debug - Model feature names: {feature_names}")
                    
                    # Create DataFrame with correct feature names
                    features_df = pd.DataFrame(features, columns=feature_names)
                    
                    # Scale features using track-specific scaler
                    features_scaled = self.model_trainer.scalers[track_type].transform(features_df)
                    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)
                    
                    # Predict
                    pred = self.model_trainer.models[track_type][action].predict(features_scaled_df)[0]
                    
                    # Clamp predictions to valid ranges
                    if action == 'steer':
                        pred = max(min(pred, 1.0), -1.0)
                    elif action in ['accel', 'brake', 'clutch']:
                        pred = max(min(pred, 1.0), 0.0)
                    else:  # gear
                        pred = round(pred)
                        pred = max(min(pred, 6), -1)  # or -1 for reverse gear
                    
                    predictions[action] = pred
                    print(f"Predicted {action}: {pred:.3f}")
                except Exception as e:
                    print(f"Error predicting {action} for {track_type} track: {e}")
                    predictions[action] = None
            else:
                print(f"Warning: No {action} model available for {track_type} track")
                predictions[action] = None
        
        return predictions
    
    def drive(self, msg):
        """Drive the car based on model predictions"""
        self.state.setFromMsg(msg)
        
        # Get current state features
        state_features = self.prepare_features_for_prediction()
        
        try:
            # Get model predictions
            predictions = self.predict_controls(state_features)
            
            # Apply model predictions
            if predictions['steer'] is not None:
                self.control.setSteer(float(predictions['steer']))
            
            if predictions['accel'] is not None:
                self.control.setAccel(float(predictions['accel']))
            
            if predictions['brake'] is not None:
                self.control.setBrake(float(predictions['brake']))
            
            if predictions['clutch'] is not None:
                self.control.setClutch(float(predictions['clutch']))
            
            if predictions['gear'] is not None:
                self.control.setGear(int(predictions['gear']))
            
            return self.control.toMsg()
            
        except Exception as e:
            print(f"Error in drive: {e}")
            # Fallback to safe default values
            self.control.setGear(1)
            self.control.setSteer(0.0)
            self.control.setAccel(0.8)
            self.control.setBrake(0.0)
            self.control.setClutch(0.0)
            return self.control.toMsg()
    
    def set_track_type(self, track_type):
        """Set the current track type and load appropriate models"""
        if track_type in self.track_type_map:
            self.current_track_type = self.track_type_map[track_type]
            print(f"Loading models for track type: {self.current_track_type}")
            self.model_trainer.load_models(self.current_track_type)
        else:
            print(f"Warning: Unknown track type {track_type}, defaulting to road")
            self.current_track_type = 'road'
            self.model_trainer.load_models('road')