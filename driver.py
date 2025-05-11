import msgParser
import carState
import carControl
import keyboard
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
        
        # Initialize feature storage for temporal features
        self.feature_history = []
        self.max_history = 2  # For lag and difference features
        
        # Define base features (matching model_trainer exactly)
        self.base_features = [
            'Angle',              # Car's angle relative to track
            'SpeedX',            # Longitudinal speed
            'SpeedY',            # Lateral speed
            'TrackPosition',      # Position relative to track center
            'RPM'                # Engine RPM
        ]
        
        # Add track sensors (matching model_trainer)
        self.base_features.extend([f'Track_{i}' for i in range(1, 20)])
        
        # Initialize control values
        self.steering_value = 0.0
        self.accel_value = 0.0
        self.brake_value = 0.0
        self.is_reverse = False
        
        # Gear shifting parameters
        self.gear_shift_rpm = {
            1: 3000,  # Shift up from 1st at 3000 RPM
            2: 3500,  # Shift up from 2nd at 3500 RPM
            3: 4000,  # Shift up from 3rd at 4000 RPM
            4: 4500,  # Shift up from 4th at 4500 RPM
            5: 5000,  # Shift up from 5th at 5000 RPM
            6: 5500   # Shift up from 6th at 5500 RPM
        }
        self.gear_down_rpm = {
            2: 2000,  # Shift down to 1st at 2000 RPM
            3: 2500,  # Shift down to 2nd at 2500 RPM
            4: 3000,  # Shift down to 3rd at 3000 RPM
            5: 3500,  # Shift down to 4th at 3500 RPM
            6: 4000   # Shift down to 5th at 4000 RPM
        }
        
        print(f"Debug - Driver initialized with stage: {stage}")
        print(f"Debug - Track mapping: {self.track_type_map}")
        
        # Set up keyboard event handlers
        keyboard.on_press_key('a', lambda _: self.handle_steering('left'))
        keyboard.on_press_key('d', lambda _: self.handle_steering('right'))
        keyboard.on_release_key('a', lambda _: self.handle_steering('left', release=True))
        keyboard.on_release_key('d', lambda _: self.handle_steering('right', release=True))
        keyboard.on_press_key('w', lambda _: self.handle_accel(True))
        keyboard.on_release_key('w', lambda _: self.handle_accel(False))
        keyboard.on_press_key('s', lambda _: self.handle_brake(True))
        keyboard.on_release_key('s', lambda _: self.handle_brake(False))
        keyboard.on_press_key('r', lambda _: self.toggle_reverse())
    
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
        
        # Define base features exactly as in model_trainer
        base_features = [
            'Angle',              # Car's angle relative to track
            'SpeedX',            # Longitudinal speed
            'SpeedY',            # Lateral speed
            'TrackPosition',      # Position relative to track center
            'RPM'                # Engine RPM
        ]
        
        # Add track sensors
        base_features.extend([f'Track_{i}' for i in range(1, 20)])
        
        # Add base features
        current_features['Angle'] = self.state.getAngle()
        current_features['SpeedX'] = self.state.getSpeedX()
        current_features['SpeedY'] = self.state.getSpeedY()
        current_features['TrackPosition'] = self.state.getTrackPos()
        current_features['RPM'] = self.state.getRpm()
        
        # Add track sensors
        track_sensors = self.state.getTrack()
        if track_sensors is not None:
            for i, value in enumerate(track_sensors, 1):
                current_features[f'Track_{i}'] = value
        
        # Convert to DataFrame
        current_df = pd.DataFrame([current_features])
        
        # Check which features are actually present
        available_features = [col for col in base_features if col in current_df.columns]
        
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
        """Predict control actions using track-specific XGBoost models"""
        predictions = {}
        
        if self.current_track_type is None:
            print("Warning: No track type detected, using default models")
            track_type = 'road'  # Default to road track
        else:
            track_type = self.current_track_type
            print(f"Using models for {track_type} track")
        
        print(f"\nDebug - Features shape before scaling: {features.shape}")
        
        for action in ['steer', 'accel', 'brake']:
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
                    
                    # Predict
                    pred = self.model_trainer.models[track_type][action].predict(features_scaled)[0]
                    
                    # Add safeguards for predictions
                    if action == 'steer':
                        # Get current track position and angle
                        track_pos = self.state.getTrackPos()
                        angle = self.state.getAngle()
                        
                        # Calculate steering correction based on track position and angle
                        position_correction = -track_pos * 0.5  # Steer more when further from center
                        angle_correction = -angle * 0.3  # Steer more when angle is larger
                        
                        # Combine model prediction with corrections
                        pred = pred + position_correction + angle_correction
                        
                        # Clamp to valid range
                        pred = max(min(pred, 1.0), -1.0)
                        
                    elif action == 'accel':
                        # Reduce acceleration when not well positioned
                        if abs(track_pos) > 0.5:
                            pred *= 0.5  # Reduce acceleration when near track edge
                        # Ensure minimum acceleration when speed is very low
                        if self.state.getSpeedX() < 5.0:
                            pred = max(pred, 0.3)  # Minimum acceleration to get moving
                        # Clamp to valid range
                        pred = max(min(pred, 1.0), 0.0)
                    else:  # brake
                        # Increase braking when not well positioned
                        if abs(track_pos) > 0.5:
                            pred = max(pred, 0.3)  # Ensure some braking when near track edge
                        # Clamp to valid range
                        pred = max(min(pred, 1.0), 0.0)
                    
                    predictions[action] = pred
                    print(f"Predicted {action}: {pred:.3f}")
                except Exception as e:
                    print(f"Error predicting {action} for {track_type} track: {e}")
                    predictions[action] = None
            else:
                print(f"Warning: No {action} model available for {track_type} track")
                predictions[action] = None
        
        return predictions
    
    def get_gear(self):
        """Determine gear based on speed and RPM"""
        speed = self.state.getSpeedX()
        rpm = self.state.getRpm()
        current_gear = self.state.getGear()
        
        # If in reverse, stay in reverse
        if current_gear < 0:
            return -1
        
        # If speed is very low, use first gear
        if speed < 5:
            return 1
        
        # Determine gear based on speed ranges
        if speed < 20:
            target_gear = 1
        elif speed < 40:
            target_gear = 2
        elif speed < 60:
            target_gear = 3
        elif speed < 80:
            target_gear = 4
        elif speed < 100:
            target_gear = 5
        else:
            target_gear = 6
        
        # RPM-based adjustments
        if rpm > self.gear_shift_rpm.get(current_gear, 6000) and current_gear < 6:
            # Shift up if RPM is too high
            target_gear = min(target_gear + 1, 6)
        elif rpm < self.gear_down_rpm.get(current_gear, 1000) and current_gear > 1:
            # Shift down if RPM is too low
            target_gear = max(target_gear - 1, 1)
        
        return target_gear
    
    def drive(self, msg):
        """Drive the car based on model predictions"""
        self.state.setFromMsg(msg)
        
        # Get current state features
        state_features = self.prepare_features_for_prediction()
        
        # Check if car is off track
        track_pos = self.state.getTrackPos()
        is_off_track = abs(track_pos) > 1.0
        
        # If off track, use reverse gear for recovery
        if is_off_track:
            print(f"Recovery mode: Off track (position: {track_pos:.2f})")
            self.control.setGear(-1)  # Reverse gear
            self.control.setSteer(0.0)  # Straight steering
            self.control.setAccel(0.5)  # Moderate throttle
            self.control.setBrake(0.0)
            return self.control.toMsg()
        
        # Normal autonomous driving
        try:
            # Get model prediction
            predictions = self.predict_controls(state_features)
            
            # Set gear based on speed
            self.control.setGear(self.get_gear())
            
            # Apply model predictions
            if predictions['steer'] is not None:
                steering = float(predictions['steer'])
                steering = max(-1.0, min(1.0, steering))
                self.control.setSteer(steering)
            
            if predictions['accel'] is not None:
                throttle = float(predictions['accel'])
                throttle = max(0.0, min(1.0, throttle))
                self.control.setAccel(throttle)
            
            if predictions['brake'] is not None:
                brake = float(predictions['brake'])
                brake = max(0.0, min(1.0, brake))
                self.control.setBrake(brake)
            
            return self.control.toMsg()
            
        except Exception as e:
            print(f"Error in drive: {e}")
            # Fallback to safe default values
            self.control.setGear(1)
            self.control.setSteer(0.0)
            self.control.setAccel(0.3)
            self.control.setBrake(0.0)
            return self.control.toMsg()
    
    def setExternalSteer(self, steer_value):
        """Set external steering input value (-1.0 to 1.0)"""
        if steer_value is not None:
            # Clamp the steering value between -1 and 1
            self.external_steer = max(min(steer_value, 1.0), -1.0)
        else:
            self.external_steer = None
    
    def setExternalAccel(self, accel_value):
        """Set external acceleration input value (0.0 to 1.0)"""
        if accel_value is not None:
            # Clamp the acceleration value between 0 and 1
            self.external_accel = max(min(accel_value, 1.0), 0.0)
        else:
            self.external_accel = None
    
    def setExternalBrake(self, brake_value):
        """Set external brake input value (0.0 to 1.0)"""
        if brake_value is not None:
            # Clamp the brake value between 0 and 1
            self.external_brake = max(min(brake_value, 1.0), 0.0)
        else:
            self.external_brake = None
    
    def steer(self):
        if self.external_steer is not None:
            # Use external steering input if available
            self.control.setSteer(self.external_steer)
        else:
            # Fall back to automatic steering if no external input
            angle = self.state.angle
            dist = self.state.trackPos
            self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def speed(self):
        # Handle external acceleration input
        if self.external_accel is not None:
            if self.is_reverse:
                # In reverse, we need to set both gear and acceleration
                self.control.setGear(-1)
                self.control.setAccel(self.external_accel)  # Don't invert acceleration
            else:
                self.control.setAccel(self.external_accel)
        else:
            # If no external acceleration, set to 0
            self.control.setAccel(0.0)
        
        # Handle external brake input
        if self.external_brake is not None:
            self.control.setBrake(self.external_brake)
        else:
            self.control.setBrake(0.0)
    
    def get_race_type(self):
        """Convert stage number to race type string"""
        if self.stage == self.WARM_UP:
            return 'warmup'
        elif self.stage == self.QUALIFYING:
            return 'qualifying'
        elif self.stage == self.RACE:
            return 'race'
        else:
            return 'unknown'
    
    def onShutDown(self):
        """Called when the race is shutting down"""
        if self.logger:
            self.logger.close()
    
    def onRestart(self):
        """Called when the race is restarting"""
        if self.logger:
            self.logger.close()
            self.logger = None
    
    def handle_steering(self, direction, release=False):
        if direction == 'left':
            if release:
                self.steering_value = 0.0
            else:
                self.steering_value = 1.0  # Changed from -1.0 to 1.0
        elif direction == 'right':
            if release:
                self.steering_value = 0.0
            else:
                self.steering_value = -1.0  # Changed from 1.0 to -1.0
        self.setExternalSteer(self.steering_value)
    
    def handle_accel(self, press):
        if press:
            self.accel_value = 1.0
        else:
            self.accel_value = 0.0
        self.setExternalAccel(self.accel_value)
    
    def handle_brake(self, press):
        if press:
            self.brake_value = 1.0
        else:
            self.brake_value = 0.0
        self.setExternalBrake(self.brake_value)
    
    def toggle_reverse(self):
        """Toggle between forward and reverse gear"""
        self.is_reverse = not self.is_reverse
        if self.is_reverse:
            # When entering reverse, ensure we're not accelerating
            self.accel_value = 0.0
            self.setExternalAccel(0.0)
            self.control.setGear(-1)
            # Set a small initial acceleration to get moving
            self.control.setAccel(0.1)
        else:
            # When exiting reverse, reset to first gear
            self.control.setGear(1)
            self.control.setAccel(0.0)
    
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