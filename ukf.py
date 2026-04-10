import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise


# ── State & measurement dimensions ───────────────────────────────────────────
#
#  State vector  x = [px, py, vx, vy]
#      px, py  : pixel position (col, row)
#      vx, vy  : pixel velocity (col/frame, row/frame)
#
#  Measurement z = [px, py]
#      We observe position only; velocity is inferred by the filter.
#
DIM_X = 4   # state dimensions
DIM_Z = 2   # measurement dimensions


# ── Transition function  (how the state evolves one frame forward) ────────────

def fx(x, dt):
    """
    Constant-velocity motion model.
    x_{t+1} = F * x_t   where F is the standard kinematic matrix.
    """
    px, py, vx, vy = x
    return np.array([
        px + vx * dt,
        py + vy * dt,
        vx,
        vy
    ])


# ── Measurement function  (what we can observe from the state) ───────────────

def hx(x):
    """Extract the pixel position from the state vector."""
    return np.array([x[0], x[1]])


# ── UKF factory ──────────────────────────────────────────────────────────────

def build_ukf(initial_pos, dt=1.0,
              process_noise_std=0.1,
              measurement_noise_std=2.0):
    """
    Construct and initialise an Unscented Kalman Filter.

    Args:
        initial_pos:           (row, col) first detected pixel position
        dt:                    time step between frames (1.0 = one frame)
        process_noise_std:     how much we trust the motion model
        measurement_noise_std: expected pixel error in centroid detection

    Returns:
        Initialised UnscentedKalmanFilter ready to call .predict() / .update()
    """
    # Sigma points — required by newer versions of filterpy
    points = MerweScaledSigmaPoints(
        n=DIM_X,
        alpha=0.1,    # spread of sigma points around the mean
        beta=2.0,     # optimal for Gaussian distributions
        kappa=0.0
    )

    ukf = UnscentedKalmanFilter(
        dim_x=DIM_X,
        dim_z=DIM_Z,
        dt=dt,
        fx=fx,
        hx=hx,
        points=points
    )

    row, col = initial_pos

    # Initial state: position from first detection, velocity unknown → 0
    ukf.x = np.array([float(col), float(row), 0.0, 0.0])

    # Initial state covariance – high uncertainty on velocity
    ukf.P = np.diag([10.0, 10.0, 100.0, 100.0])

    # Process noise (model uncertainty)
    q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_std**2)
    ukf.Q = np.block([
        [q,               np.zeros((2, 2))],
        [np.zeros((2, 2)), q              ]
    ])

    # Measurement noise – isotropic position error in pixels
    ukf.R = np.eye(DIM_Z) * measurement_noise_std**2

    return ukf


# ── Convenience wrapper ───────────────────────────────────────────────────────

class SatelliteTracker:
    """
    Thin wrapper around the UKF that handles initialisation on the first
    detection and exposes a simple update() interface.

    Usage:
        tracker = SatelliteTracker()
        for centroid in detections:
            state = tracker.update(centroid)
            print(state)   # [px, py, vx, vy]
    """

    def __init__(self, dt=1.0,
                 process_noise_std=0.1,
                 measurement_noise_std=2.0):
        self.dt  = dt
        self.pns = process_noise_std
        self.mns = measurement_noise_std
        self.ukf = None

    def update(self, centroid):
        """
        Feed one centroid measurement into the filter.

        Args:
            centroid: (row, col) pixel coordinates, or None if no detection
                      this frame (filter will still predict forward)

        Returns:
            np.array [px, py, vx, vy] – current state estimate, or None
        """
        if self.ukf is None:
            if centroid is None:
                return None             # nothing to initialise from yet
            self.ukf = build_ukf(
                initial_pos=centroid,
                dt=self.dt,
                process_noise_std=self.pns,
                measurement_noise_std=self.mns
            )
            return self.ukf.x.copy()

        self.ukf.predict()

        if centroid is not None:
            row, col = centroid
            z = np.array([float(col), float(row)])
            self.ukf.update(z)

        return self.ukf.x.copy()

    @property
    def state(self):
        """Current [px, py, vx, vy] estimate, or None before first detection."""
        return self.ukf.x.copy() if self.ukf is not None else None
