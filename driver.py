import msgParser
import carState
import carControl
import keyboard
import time
import numpy as np
from data_logger import DataLogger
from model_trainer import TORCSModelTrainer

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
        
        # Simplified steering and speed parameters
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        
        # Add external control inputs
        self.external_steer = None
        self.external_accel = None
        self.external_brake = None
        
        # Initialize keyboard controls
        self.steering_value = 0.0
        self.accel_value = 0.0
        self.brake_value = 0.0
        self.is_reverse = False
        
        # Initialize data logger
        self.logger = None
        
        # Initialize model trainer and load models
        self.model_trainer = TORCSModelTrainer()
        self.model_trainer.load_models()
        
        # Track-specific parameters
        self.track_params = {
            'G-Speedway': {'max_speed': 120, 'steer_lock': 0.785398},  # Oval track
            'E-Track3': {'max_speed': 100, 'steer_lock': 0.785398},    # Road track
            'Dirt2': {'max_speed': 80, 'steer_lock': 0.785398}         # Dirt track
        }
        
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
        """Prepare current state features for model prediction"""
        features = [
            self.state.getSpeedX(),
            self.state.getSpeedY(),
            self.state.getSpeedZ(),
            self.state.getRpm(),
            self.state.getGear(),
            self.state.getAngle(),
            self.state.getTrackPos(),
            self.state.getTrackEdgeDist()
        ]
        
        # Add track sensors
        features.extend(self.state.getTrack())
        
        # Add opponent sensors
        features.extend(self.state.getOpponents())
        
        return np.array(features).reshape(1, -1)
    
    def predict_controls(self, features):
        """Predict control actions using XGBoost models"""
        predictions = {}
        
        for action in ['steer', 'accel', 'brake']:
            if self.model_trainer.models[action] is not None:
                try:
                    # Scale features
                    features_scaled = self.model_trainer.scalers[action].transform(features)
                    # Predict
                    pred = self.model_trainer.models[action].predict(features_scaled)[0]
                    # Clamp predictions to valid ranges
                    if action == 'steer':
                        pred = max(min(pred, 1.0), -1.0)
                    else:  # accel and brake
                        pred = max(min(pred, 1.0), 0.0)
                    predictions[action] = pred
                except Exception as e:
                    print(f"Error predicting {action}: {e}")
                    predictions[action] = None
            else:
                predictions[action] = None
        
        return predictions
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        # Update track-specific parameters if track name is available
        if hasattr(self.state, 'getTrackName'):
            track_name = self.state.getTrackName()
            if track_name in self.track_params:
                params = self.track_params[track_name]
                self.max_speed = params['max_speed']
                self.steer_lock = params['steer_lock']
        
        # Try to use model predictions if available
        if not any([self.external_steer, self.external_accel, self.external_brake]):
            features = self.prepare_features_for_prediction()
            predictions = self.predict_controls(features)
            
            if predictions['steer'] is not None:
                self.control.setSteer(predictions['steer'])
            else:
                self.steer()
            
            if predictions['accel'] is not None:
                self.control.setAccel(predictions['accel'])
            else:
                self.speed()
            
            if predictions['brake'] is not None:
                self.control.setBrake(predictions['brake'])
            else:
                self.speed()
        else:
            # Use manual/external controls if available
            self.steer()
            self.gear()
            self.speed()
        
        # Log data if logger is initialized
        if self.logger:
            self.logger.log_data(self.state, self.control, 
                               self.state.getTrackName() if hasattr(self.state, 'getTrackName') else 'unknown',
                               self.get_race_type())
        
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
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = self.state.getSpeedX()
        
        # Handle reverse gear
        if self.is_reverse:
            self.control.setGear(-1)
            return
        
        # More aggressive downshifting based on speed and RPM
        if speed < 10:  # Lower speed threshold for downshifting
            gear = 1
        elif speed < 20 and rpm < 4000:  # Downshift at lower RPM when speed is low
            gear = max(1, gear - 1)
        elif speed < 30 and rpm < 3500:
            gear = max(1, gear - 1)
        elif speed < 40 and rpm < 3000:
            gear = max(1, gear - 1)
        else:
            # Normal upshifting logic
            if self.prev_rpm == None:
                up = True
            else:
                if (self.prev_rpm - rpm) < 0:
                    up = True
                else:
                    up = False
            
            if up and rpm > 7000:
                gear += 1
            
            if not up and rpm < 3000:
                gear -= 1
            
            # Ensure gear stays within valid range
            gear = max(1, min(gear, 6))
        
        self.control.setGear(gear)
    
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