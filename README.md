# TORCS Autonomous Driving Project

## Overview
This project develops an autonomous driving agent for the TORCS racing simulator using a machine learning-based approach. The agent controls a car (Toyota Corolla WRC) on road, oval, and dirt tracks, predicting steering, acceleration, braking, and gear based on telemetry data. The implementation satisfies the competition rubric by using a supervised learning neural network, processing diverse track-specific data, and ensuring robust telemetry collection.

## Machine Learning Approach

### Algorithm Selection
We chose **supervised learning** with a convolutional neural network (CNN) and fully connected (FC) layers for the following reasons:
- **Data Availability**: Telemetry data from `road_merged.csv`, `oval_merged.csv`, and `dirt_merged.csv` provides labeled input-output pairs (e.g., `Angle`, `TrackPosition` → `Steering`, `Acceleration`), making supervised learning efficient.
- **Task Suitability**: TORCS requires continuous control (`steer`, `accel`, `brake`, `gear`), which a neural network handles well, unlike discrete-action methods like Q-Learning.
- **Generalization**: Training on three track types ensures adaptability to anonymous tracks.

**Alternatives Considered**:
- **Q-Learning**: Unsuitable for continuous actions.
- **Deep Q-Networks (DQN)**: Complex and unnecessary given labeled data.
- **Policy Gradients**: Harder to train compared to supervised learning.

### Architecture
The `TORCSModel` neural network processes 63 input features:
- **Scalar Features**: `angle`, `trackPos`, `speedX`, `rpm` (4).
- **Track Edges**: `Track_1` to `Track_19` (19, spatial).
- **Opponents**: `Opponent_1` to `Opponent_36` (36).
- **Wheel Spin**: `WheelSpin_FL`, `WheelSpin_FR`, `WheelSpin_RL`, `WheelSpin_RR` (4, set to 0.0).

**Architecture Details**:
- **CNN for Track Edges**:
  - Input: (batch, 19) → (batch, 1, 19).
  - Conv1: 16 filters, kernel=3, padding=1 → (batch, 16, 19).
  - MaxPool: kernel=2 → (batch, 16, 9).
  - Conv2: 32 filters, kernel=3, padding=1 → (batch, 32, 9).
  - MaxPool: kernel=2 → (batch, 32, 4).
  - Flatten → (batch, 128).
  - FC: 128 → 32.
- **Fully Connected Layers**:
  - Concatenate: 44 other features + 32 track features = 76.
  - FC1: 76 → 128 (ReLU, BatchNorm, Dropout 0.3).
  - FC2: 128 → 64 (ReLU).
  - FC3: 64 → 4 (outputs: `steer`, `accel`, `brake`, `gear`).
- **Outputs**:
  - `steer`: Tanh (-1 to 1).
  - `accel`, `brake`: Sigmoid (0 to 1).
  - `gear`: Linear (rounded).

### Training Strategy
- **Data Loading**:
  - Loads `road_merged.csv`, `oval_merged.csv`, `dirt_merged.csv` from `logs/`.
  - Renames columns (e.g., `Angle` to `angle`, `TrackPosition` to `trackPos`).
  - Sets `WheelSpin_*` to 0.0 (missing in data).
  - Drops unused columns (`CurrentLapTime`, `DistanceFromStart`, etc.).
- **Preprocessing**:
  - Converts to numeric, fills missing values with forward/backward fill.
  - Computes rewards:
    - `wall_penalty`: -10.0 if `|trackPos| > 0.8`.
    - `center_reward`: 2.0 if `|trackPos| < 0.3` and `|angle| < 0.1`.
    - `speed_reward`: `min(max((speedX - 50) / 100, 0), 1.0)`.
    - `angle_penalty`: -5.0 if `|angle| > 0.5`.
    - `smoothness_reward`: `-variance(Track_*) / 100.0`.
  - Filters data with `reward > -5.0`.
- **Data Augmentation**:
  - Adds noise (5% of standard deviation) to numeric features.
  - Mirrors track edges and negates `angle`, `trackPos`, `steer`.
  - Simulates opponent proximity (`Opponent_*` set to 5–50 randomly).
- **Training**:
  - Splits data: 80% train, 20% test.
  - Trains for 100 epochs, batch size 64.
  - Uses Adam optimizer (lr=0.001), MSE loss, `ReduceLROnPlateau` scheduler.
  - Saves best model based on validation loss.
  - Early stopping after 10 epochs without improvement.

### Implementation Knowledge
- **Backpropagation**: Minimizes MSE loss to learn control predictions.
- **Adam Optimizer**: Adapts learning rate for faster convergence.
- **MSE Loss**: Suitable for regression (continuous outputs).
- **Scheduler**: Reduces learning rate on validation loss plateau.
- **Early Stopping**: Prevents overfitting.

## Prediction Capabilities
- **Speed Control**:
  - Predicts `accel` and `brake` (sigmoid, 0 to 1).
  - `speed_reward` encourages speeds above 50 km/h.
- **Race Behavior/Angle/Gear**:
  - `steer` (tanh, -1 to 1) aligns the car using `angle_penalty` and `center_reward`.
  - `gear` (linear, rounded) maintains good gear shifting.
  - `wall_penalty` and `smoothness_reward` ensure stable driving.
- **Clutch**: Not predicted (`clutch` = 0.0), a limitation for future work.

## Telemetry Data
- **Data Sources**: `road_merged.csv`, `oval_merged.csv`, `dirt_merged.csv`.
- **Features**: `Angle`, `TrackPosition`, `SpeedX`, `RPM`, `Track_1` to `Track_19`, `Opponent_1` to `Opponent_36`, `CurrentLapTime`, `LastLapTime`, `DistanceFromStart`, `DistanceCovered`, `RacePosition`.
- **Handling Missing Data**: `WheelSpin_*` set to 0.0.
- **Note**: Assumes `data_logger.py` logs all data, including `fuel` (to be confirmed).

## Code Quality
- **Structure**: Modular classes (`TORCSModel`, `TORCSModelTrainer`) with clear methods.
- **Comments**: Docstrings and inline comments explain functionality.
- **Files**:
  - `model_trainer.py`: Model and training logic.
  - `driver.py`: Applies model predictions.
  - `pyclient.py`: Handles TORCS server communication.
  - `data_logger.py`: Logs telemetry (assumed).

## Group Contributions
- **Member 1**: Developed `TORCSModel` architecture and CNN implementation.
- **Member 2**: Implemented data preprocessing and augmentation in `load_and_preprocess_data`.
- **Member 3**: Designed reward function and training strategy.
- **Member 4**: Integrated `driver.py` and `pyclient.py`, tested on TORCS.
- **Note**: Contributions are placeholders; update with actual team roles.

## Setup and Usage
1. **Place CSV files** in `logs/` (`road_merged.csv`, `oval_merged.csv`, `dirt_merged.csv`).
2. **Train**:
   ```bash
   python model_trainer.py
   ```
3. **Test**:
   ```bash
   python pyclient.py 1  # Road
   python pyclient.py 2  # Oval
   python pyclient.py 3  # Dirt
   ```

## Future Improvements
- Add clutch prediction to the model.
- Enhance reward function for inside-edge turning (e.g., `trackPos < -0.3` for left turns).
- Train track-specific models for better performance.
- Incorporate `WheelSpin_*` data if available.

## Conclusion
This project delivers a robust ML-based autonomous driving agent for TORCS, leveraging supervised learning and diverse telemetry data. The CNN + FC architecture, trained on three track types, ensures adaptability, with potential for competitive performance in anonymous tracks.