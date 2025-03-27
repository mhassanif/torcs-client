import csv
import time
from datetime import datetime
import os

class DataLogger:
    def __init__(self, track_name, race_type):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Use a single file for all races
        self.filename = 'logs/race_data.csv'
        
        # Define CSV headers
        self.headers = [
            # Race information
            'timestamp', 'lap_number', 'lap_time', 'race_position',
            
            # Car state
            'speed_x', 'speed_y', 'speed_z', 'rpm', 'gear', 'fuel',
            'angle', 'track_position', 'track_edge_dist',
            
            # Track sensors (19 values)
            'track_sensor_0', 'track_sensor_1', 'track_sensor_2', 'track_sensor_3',
            'track_sensor_4', 'track_sensor_5', 'track_sensor_6', 'track_sensor_7',
            'track_sensor_8', 'track_sensor_9', 'track_sensor_10', 'track_sensor_11',
            'track_sensor_12', 'track_sensor_13', 'track_sensor_14', 'track_sensor_15',
            'track_sensor_16', 'track_sensor_17', 'track_sensor_18',
            
            # Opponent sensors (36 values)
            'opponent_sensor_0', 'opponent_sensor_1', 'opponent_sensor_2', 'opponent_sensor_3',
            'opponent_sensor_4', 'opponent_sensor_5', 'opponent_sensor_6', 'opponent_sensor_7',
            'opponent_sensor_8', 'opponent_sensor_9', 'opponent_sensor_10', 'opponent_sensor_11',
            'opponent_sensor_12', 'opponent_sensor_13', 'opponent_sensor_14', 'opponent_sensor_15',
            'opponent_sensor_16', 'opponent_sensor_17', 'opponent_sensor_18', 'opponent_sensor_19',
            'opponent_sensor_20', 'opponent_sensor_21', 'opponent_sensor_22', 'opponent_sensor_23',
            'opponent_sensor_24', 'opponent_sensor_25', 'opponent_sensor_26', 'opponent_sensor_27',
            'opponent_sensor_28', 'opponent_sensor_29', 'opponent_sensor_30', 'opponent_sensor_31',
            'opponent_sensor_32', 'opponent_sensor_33', 'opponent_sensor_34', 'opponent_sensor_35',
            
            # Car control inputs
            'accel', 'brake', 'steer', 'clutch',
            
            # Race metadata
            'track_name', 'race_type', 'damage', 'distance_from_start', 'distance_raced',
            
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
        
        # Generate a unique session ID for this race
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def log_data(self, car_state, car_control, track_name, race_type):
        current_time = time.time() - self.start_time
        
        # Calculate lap number and time
        if car_state.getLastLapTime() != self.last_lap_time:
            self.current_lap += 1
            self.last_lap_time = car_state.getLastLapTime()
        
        # Prepare row data
        row_data = [
            # Race information
            current_time, self.current_lap, car_state.getLastLapTime(), car_state.getRacePos(),
            
            # Car state
            car_state.getSpeedX(), car_state.getSpeedY(), car_state.getSpeedZ(),
            car_state.getRpm(), car_state.getGear(), car_state.getFuel(),
            car_state.getAngle(), car_state.getTrackPos(), car_state.getTrackEdgeDist(),
            
            # Track sensors
            *car_state.getTrack(),
            
            # Opponent sensors
            *car_state.getOpponents(),
            
            # Car control inputs
            car_control.getAccel(), car_control.getBrake(), car_control.getSteer(), car_control.getClutch(),
            
            # Race metadata
            track_name, race_type, car_state.getDamage(),
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