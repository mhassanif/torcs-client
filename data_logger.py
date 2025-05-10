import csv
import time
from datetime import datetime
import os

class DataLogger:
    def __init__(self, track_name, race_type, car_type='unknown'):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Use a single file for all races
        self.filename = 'logs/race_data.csv'
        
        # Define CSV headers - simplified based on new scope
        self.headers = [
            # Race information
            'timestamp', 'lap_number', 'lap_time', 'race_position',
            
            # Car state (removed fuel and damage)
            'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 'Gear',
            'Angle', 'TrackPosition', 'track_edge_dist',
            
            # Track sensors (19 values) - 1-based indexing
            'Track_1', 'Track_2', 'Track_3', 'Track_4',
            'Track_5', 'Track_6', 'Track_7', 'Track_8',
            'Track_9', 'Track_10', 'Track_11', 'Track_12',
            'Track_13', 'Track_14', 'Track_15', 'Track_16',
            'Track_17', 'Track_18', 'Track_19',
            
            # Opponent sensors (36 values) - 1-based indexing
            'Opponent_1', 'Opponent_2', 'Opponent_3', 'Opponent_4',
            'Opponent_5', 'Opponent_6', 'Opponent_7', 'Opponent_8',
            'Opponent_9', 'Opponent_10', 'Opponent_11', 'Opponent_12',
            'Opponent_13', 'Opponent_14', 'Opponent_15', 'Opponent_16',
            'Opponent_17', 'Opponent_18', 'Opponent_19', 'Opponent_20',
            'Opponent_21', 'Opponent_22', 'Opponent_23', 'Opponent_24',
            'Opponent_25', 'Opponent_26', 'Opponent_27', 'Opponent_28',
            'Opponent_29', 'Opponent_30', 'Opponent_31', 'Opponent_32',
            'Opponent_33', 'Opponent_34', 'Opponent_35', 'Opponent_36',
            
            # Car control inputs
            'Acceleration', 'Braking', 'Steering', 'Clutch',
            
            # Race metadata (removed damage and fuel)
            'track_name', 'race_type', 'car_type', 'DistanceFromStart', 'DistanceCovered',
            
            # Race session info
            'session_id', 'session_start_time'
        ]
        
        # Initialize CSV file with headers only if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        
        self.start_time = time.time()
        self.last_lap_time = 0
        self.current_lap = 0
        self.car_type = car_type
        
        # Generate a unique session ID for this race
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def log_data(self, car_state, car_control, track_name, race_type):
        current_time = time.time() - self.start_time
        
        # Calculate lap number and time
        if car_state.getLastLapTime() != self.last_lap_time:
            self.current_lap += 1
            self.last_lap_time = car_state.getLastLapTime()
        
        # Get sensor arrays
        track_sensors = car_state.getTrack()
        opponent_sensors = car_state.getOpponents()
        
        # Prepare row data - simplified based on new scope
        row_data = [
            # Race information
            current_time, self.current_lap, car_state.getLastLapTime(), car_state.getRacePos(),
            
            # Car state (removed fuel and damage)
            car_state.getSpeedX(), car_state.getSpeedY(), car_state.getSpeedZ(),
            car_state.getRpm(), car_state.getGear(),
            car_state.getAngle(), car_state.getTrackPos(), car_state.getTrackEdgeDist(),
            
            # Track sensors - convert to 1-based indexing
            *(track_sensors if track_sensors else [0] * 19),
            
            # Opponent sensors - convert to 1-based indexing
            *(opponent_sensors if opponent_sensors else [0] * 36),
            
            # Car control inputs
            car_control.getAccel(), car_control.getBrake(), car_control.getSteer(), car_control.getClutch(),
            
            # Race metadata (removed damage and fuel)
            track_name, race_type, self.car_type,
            car_state.getDistFromStart(), car_state.getDistRaced(),
            
            # Session info
            self.session_id, self.session_start_time
        ]
        
        # Append to CSV
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    
    def close(self):
        """Close the logger and save any remaining data"""
        pass  # CSV file is automatically closed after each write 