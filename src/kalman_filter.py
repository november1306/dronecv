import numpy as np


class KalmanFilter:
    def __init__(self, dt=1.0, state_uncertainty=1e-5, measurement_uncertainty=1e-1, initial_state=None):
        self.dt = dt
        self.state_uncertainty = state_uncertainty
        self.measurement_uncertainty = measurement_uncertainty

        # State transition matrix
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance
        self.Q = np.eye(4) * state_uncertainty

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_uncertainty

        # Initial state covariance
        self.P = np.eye(4)

        # Initial state
        self.x = np.zeros((4, 1)) if initial_state is None else initial_state

    def predict(self):
        # Predict the state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:2].flatten()

    def update(self, measurement):
        # Update the state based on the measurement
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[:2].flatten()

    def get_state(self):
        return self.x[:2].flatten()
