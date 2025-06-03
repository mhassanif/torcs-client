import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple

class TORCSModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(TORCSModel, self).__init__()
        # CNN for track edge sensors
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc_track = nn.Linear(32 * 4, 32)  # After pooling: 19 -> 9 -> 4

        # Fully connected layers for all features
        self.fc1 = nn.Linear(input_size - 19 + 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 4)  # Outputs: steer, accel, brake, gear
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor, track_edges: torch.Tensor) -> torch.Tensor:
        print(f"Input x (other_features) shape: {x.shape}")  # Debug
        print(f"Input track_edges shape: {track_edges.shape}")  # Debug

        # Process track edges with CNN
        track_edges = track_edges.unsqueeze(1)  # Add channel dimension: (batch, 1, 19)
        track_edges = self.relu(self.conv1(track_edges))  # (batch, 16, 19)
        track_edges = self.pool(track_edges)  # (batch, 16, 9)
        track_edges = self.relu(self.conv2(track_edges))  # (batch, 32, 9)
        track_edges = self.pool(track_edges)  # (batch, 32, 4)
        track_edges = track_edges.view(track_edges.size(0), -1)  # (batch, 128)
        track_edges = self.relu(self.fc_track(track_edges))  # (batch, 32)
        print(f"Processed track_edges shape: {track_edges.shape}")  # Debug

        # Combine with other features
        x = torch.cat((x, track_edges), dim=1)  # (batch, 44 + 32 = 76)
        print(f"Combined features shape: {x.shape}")  # Debug

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply activation functions for outputs
        steer = torch.tanh(x[:, 0])  # -1 to 1
        accel = torch.sigmoid(x[:, 1])  # 0 to 1
        brake = torch.sigmoid(x[:, 2])  # 0 to 1
        gear = x[:, 3]  # Linear output for gear (will be rounded)

        return torch.stack([steer, accel, brake, gear], dim=1)

class TORCSModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = "models/torcs_model.pth"
        self.scaler_path = "models/scaler.joblib"
        os.makedirs("models", exist_ok=True)
        print(f"TORCS Model Trainer initialized on {self.device}")

    def compute_reward(self, row: pd.Series) -> float:
        """Compute reward to filter good driving behaviors"""
        track_pos = row['trackPos']
        speed_x = row['speedX']
        angle = row['angle']
        track_edges = [row[f'Track_{i}'] for i in range(1, 20)]

        wall_penalty = -10.0 if abs(track_pos) > 0.8 else 0.0
        center_reward = 2.0 if abs(track_pos) < 0.3 and abs(angle) < 0.1 else 0.0
        speed_reward = min(max((speed_x - 50) / 100, 0), 1.0)
        angle_penalty = -5.0 if abs(angle) > 0.5 else 0.0
        edge_variance = np.var(track_edges)
        smoothness_reward = -edge_variance / 100.0

        return wall_penalty + center_reward + speed_reward + angle_penalty + smoothness_reward

    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment data to improve robustness"""
        augmented_dfs = [df]

        noise_df = df.copy()
        numeric_cols = ['angle', 'trackPos', 'speedX', 'rpm'] + \
                       [f'Track_{i}' for i in range(1, 20)] + \
                       [f'Opponent_{i}' for i in range(1, 37)] + \
                       ['WheelSpin_FL', 'WheelSpin_FR', 'WheelSpin_RL', 'WheelSpin_RR']
        for col in numeric_cols:
            if col in noise_df.columns:
                noise = np.random.normal(0, 0.05 * noise_df[col].std(), size=len(noise_df))
                noise_df[col] = noise_df[col] + noise
        augmented_dfs.append(noise_df)

        mirror_df = df.copy()
        for i in range(1, 10):
            mirror_df[f'Track_{i}'], mirror_df[f'Track_{20-i}'] = \
                df[f'Track_{20-i}'], df[f'Track_{i}']
        mirror_df['angle'] = -df['angle']
        mirror_df['trackPos'] = -df['trackPos']
        mirror_df['steer'] = -df['steer']
        augmented_dfs.append(mirror_df)

        opponent_df = df.copy()
        for i in range(1, 37):
            if np.random.random() < 0.1:
                opponent_df[f'Opponent_{i}'] = np.random.uniform(5, 50)
        augmented_dfs.append(opponent_df)

        return pd.concat(augmented_dfs, ignore_index=True)

    def load_and_preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess CSV data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {len(df)} rows")

        for col in ['track_edges', 'opponents', 'wheel_spin']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
                if col == 'track_edges':
                    for i in range(19):
                        df[f'Track_{i+1}'] = df[col].apply(lambda x: x[i] if i < len(x) else 0.0)
                elif col == 'opponents':
                    for i in range(36):
                        df[f'Opponent_{i+1}'] = df[col].apply(lambda x: x[i] if i < len(x) else 200.0)
                elif col == 'wheel_spin':
                    wheels = ['FL', 'FR', 'RL', 'RR']
                    for i, wheel in enumerate(wheels):
                        df[f'WheelSpin_{wheel}'] = df[col].apply(lambda x: x[i] if i < len(x) else 0.0)
                df = df.drop(col, axis=1)

        df = df.drop(['timestamp', 'car', 'lap', 'distance', 'track'], axis=1, errors='ignore')
        numeric_cols = df.columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()

        df['reward'] = df.apply(self.compute_reward, axis=1)
        df = df[df['reward'] > -5.0]
        print(f"Rows after reward filtering: {len(df)}")

        df = self.augment_data(df)
        print(f"Data shape after augmentation: {df.shape}")

        return df

    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare features and targets"""
        feature_columns = [
            'angle', 'trackPos', 'speedX', 'rpm',
            *[f'Track_{i}' for i in range(1, 20)],
            *[f'Opponent_{i}' for i in range(1, 37)],
            'WheelSpin_FL', 'WheelSpin_FR', 'WheelSpin_RL', 'WheelSpin_RR'
        ]
        target_columns = ['steer', 'accel', 'brake', 'gear']

        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Available features: {available_features}")
        print(f"Number of features: {len(available_features)}")

        X = df[available_features].values
        y = df[target_columns].values

        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.scaler_path)

        return X_scaled, y, available_features

    def train_model(self, csv_path: str, epochs: int = 100, batch_size: int = 64):
        """Train the model"""
        df = self.load_and_preprocess_data(csv_path)
        X, y, feature_names = self.prepare_features_and_targets(df)

        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature names: {feature_names}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        self.model = TORCSModel(input_size=len(feature_names)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.MSELoss()

        track_edge_indices = [feature_names.index(f'Track_{i}') for i in range(1, 20)]
        other_feature_indices = [i for i in range(len(feature_names)) if i not in track_edge_indices]

        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]

                track_edges = batch_X[:, track_edge_indices]
                other_features = batch_X[:, other_feature_indices]
                print(f"other_features shape: {other_features.shape}")  # Should be (batch, 44)
                print(f"track_edges shape: {track_edges.shape}")  # Should be (batch, 19)

                self.optimizer.zero_grad()
                outputs = self.model(other_features, track_edges)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                track_edges_test = X_test_tensor[:, track_edge_indices]
                other_features_test = X_test_tensor[:, other_feature_indices]
                val_outputs = self.model(other_features_test, track_edges_test)
                val_loss = criterion(val_outputs, y_test_tensor)

            self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Saved best model with val loss: {val_loss.item():.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    def load_model(self):
        """Load trained model and scaler"""
        self.model = TORCSModel(input_size=63).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.scaler = joblib.load(self.scaler_path)
        print("Model and scaler loaded")

if __name__ == '__main__':
    trainer = TORCSModelTrainer()
    trainer.train_model('logs/data.csv')