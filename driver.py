import msgParser
import carState
import carControl
import pandas as pd
import numpy as np
import torch
from model_trainer import TORCSModel, TORCSModelTrainer

class Driver(object):
    def __init__(self, stage, car_type='unknown'):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.car_type = car_type
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.trainer = TORCSModelTrainer()
        self.trainer.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Driver initialized with stage: {stage}")

    def init(self):
        """Return init string with rangefinder angles"""
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})

    def prepare_features(self):
        """Prepare features for model prediction"""
        features = {}
        feature_columns = [
            'angle', 'trackPos', 'speedX', 'rpm',
            *[f'Track_{i}' for i in range(1, 20)],
            *[f'Opponent_{i}' for i in range(1, 37)],
            'WheelSpin_FL', 'WheelSpin_FR', 'WheelSpin_RL', 'WheelSpin_RR'
        ]

        # State features
        features['angle'] = self.state.getAngle()
        features['trackPos'] = self.state.getTrackPos()
        features['speedX'] = self.state.getSpeedX()
        features['rpm'] = self.state.getRpm()

        # Track sensors
        track_sensors = self.state.getTrack()
        if track_sensors is not None:
            for i, value in enumerate(track_sensors, 1):
                features[f'Track_{i}'] = value

        # Opponent sensors
        opponents = self.state.getOpponents()
        if opponents is not None:
            for i, value in enumerate(opponents, 1):
                features[f'Opponent_{i}'] = value

        # Wheel spin
        wheel_spin = self.state.getWheelSpinVel()
        if wheel_spin is not None:
            wheels = ['FL', 'FR', 'RL', 'RR']
            for i, wheel in enumerate(wheels):
                features[f'WheelSpin_{wheel}'] = wheel_spin[i] if i < len(wheel_spin) else 0.0

        # Convert to DataFrame
        df = pd.DataFrame([features])
        available_features = [col for col in feature_columns if col in df.columns]
        df = df[available_features].astype(float).ffill().bfill()

        # Scale features
        X_scaled = self.trainer.scaler.transform(df.values)
        return X_scaled, df[available_features].columns

    def drive(self, msg):
        """Drive based on model predictions"""
        self.state.setFromMsg(msg)
        try:
            # Prepare features
            X_scaled, feature_names = self.prepare_features()
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            track_edges = X_tensor[:, 4:23]  # Track_1 to Track_19
            other_features = torch.cat((X_tensor[:, :4], X_tensor[:, 23:]), dim=1)

            # Predict
            self.trainer.model.eval()
            with torch.no_grad():
                predictions = self.trainer.model(other_features, track_edges).cpu().numpy()[0]

            # Apply predictions
            self.control.setSteer(float(predictions[0]))  # steer
            self.control.setAccel(float(predictions[1]))  # accel
            self.control.setBrake(float(predictions[2]))  # brake
            self.control.setGear(int(round(predictions[3])))  # gear, rounded
            self.control.setClutch(0.0)  # clutch not modeled

            return self.control.toMsg()

        except Exception as e:
            print(f"Error in drive: {e}")
            # Fallback to safe defaults
            self.control.setGear(1)
            self.control.setSteer(0.0)
            self.control.setAccel(0.8)
            self.control.setBrake(0.0)
            self.control.setClutch(0.0)
            return self.control.toMsg()

    def set_track_type(self, track_type):
        """Set track type (not used since model is generic)"""
        print(f"Track type set to {track_type}, using generic model")