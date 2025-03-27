import csv
from datetime import datetime
import os

class DataLogger:
    def __init__(self, filename='telemetry_data.csv'):
        self.filename = filename
        self.file = None
        self.writer = None
        self.headers_written = False
        
        # Create directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        self.filepath = os.path.join('logs', self.filename)
        
    def start_logging(self):
        """Initialize the CSV file and writer"""
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.headers_written = False
        
    def log_data(self, data_dict):
        """Log a dictionary of data to CSV"""
        if not self.file:
            self.start_logging()
            
        # Write headers if not already written
        if not self.headers_written:
            headers = list(data_dict.keys())
            self.writer.writerow(headers)
            self.headers_written = True
            
        # Write data row
        self.writer.writerow(data_dict.values())
        self.file.flush()  # Ensure data is written immediately
        
    def stop_logging(self):
        """Clean up logging resources"""
        if self.file:
            self.file.close()
            self.file = None
        self.headers_written = False

    def get_telemetry_data(self, state, control):
        """Extract and format all telemetry data"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'speedX': state.getSpeedX(),
            'speedY': state.getSpeedY(),
            'speedZ': state.getSpeedZ(),
            'rpm': state.getRpm(),
            'gear': state.getGear(),
            'angle': state.getAngle(),
            'trackPos': state.getTrackPos(),
            'damage': state.getDamage(),
            'distFromStart': state.getDistFromStart(),
            'distRaced': state.getDistRaced(),
            'racePos': state.getRacePos(),
            'accel': control.getAccel(),
            'brake': control.getBrake(),
            'steer': control.getSteer(),
            'clutch': control.getClutch(),
            'fuel': state.getFuel(),
            'curLapTime': state.getCurLapTime(),
            'lastLapTime': state.getLastLapTime(),
            'wheelSpinVel_FL': state.getWheelSpinVel()[0] if state.getWheelSpinVel() else None,
            'wheelSpinVel_FR': state.getWheelSpinVel()[1] if state.getWheelSpinVel() else None,
            'wheelSpinVel_RL': state.getWheelSpinVel()[2] if state.getWheelSpinVel() else None,
            'wheelSpinVel_RR': state.getWheelSpinVel()[3] if state.getWheelSpinVel() else None,
            'z': state.getZ()
        }
        
        # Add track sensors
        track = state.getTrack()
        if track:
            for i in range(min(19, len(track))):
                data[f'track_{i}'] = track[i]
        
        # Add opponent sensors
        opponents = state.getOpponents()
        if opponents:
            for i in range(min(36, len(opponents))):
                data[f'opponent_{i}'] = opponents[i]
        
        return data