import msgParser
import carState
import carControl
import keyboard
import time
from datetime import datetime
from dataLogger import DataLogger

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
       
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.logger = DataLogger()  # Initialize data logger
        
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
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        self.steer()
        self.gear()
        self.speed()
        
        # Log telemetry data
        telemetry_data = {
            'timestamp': datetime.now().isoformat(),
            'speedX': self.state.speedX,
            'speedY': self.state.speedY,
            'speedZ': self.state.speedZ,
            'rpm': self.state.rpm,
            'gear': self.state.gear,
            'angle': self.state.angle,
            'trackPos': self.state.trackPos,
            'damage': self.state.damage,
            'distFromStart': self.state.distFromStart,
            'distRaced': self.state.distRaced,
            'racePos': self.state.racePos,
            'accel': self.control.getAccel(),
            'brake': self.control.getBrake(),
            'steer': self.control.getSteer(),
            'clutch': self.control.getClutch(),
            'fuel': self.state.fuel,
            'curLapTime': self.state.curLapTime,
            'lastLapTime': self.state.lastLapTime,
            'z': self.state.z
        }
        
        # Add track sensors
        if hasattr(self.state, 'track') and self.state.track is not None:
            for i in range(min(19, len(self.state.track))):
                telemetry_data[f'track_{i}'] = self.state.track[i]
        
        # Add opponent sensors
        if hasattr(self.state, 'opponents') and self.state.opponents is not None:
            for i in range(min(36, len(self.state.opponents))):
                telemetry_data[f'opponent_{i}'] = self.state.opponents[i]
        
        # Add wheel spin velocities
        if hasattr(self.state, 'wheelSpinVel') and self.state.wheelSpinVel is not None and len(self.state.wheelSpinVel) >= 4:
            telemetry_data['wheelSpinVel_FL'] = self.state.wheelSpinVel[0]
            telemetry_data['wheelSpinVel_FR'] = self.state.wheelSpinVel[1]
            telemetry_data['wheelSpinVel_RL'] = self.state.wheelSpinVel[2]
            telemetry_data['wheelSpinVel_RR'] = self.state.wheelSpinVel[3]
        
        self.logger.log_data(telemetry_data)
        
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
    
    def onShutDown(self):
        self.logger.stop_logging()
    
    def onRestart(self):
        self.logger.stop_logging()
        self.logger.start_logging()

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