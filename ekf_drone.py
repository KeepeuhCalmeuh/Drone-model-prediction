"""
Improved EKF for drone inertial tracking.

Main fixes over the original version:
- Uses IMU acceleration as CONTROL INPUT instead of state overwrite
- Uses gyroscope to propagate orientation
- Rotates body acceleration into world frame
- Removes gravity correctly
- Uses quaternion attitude
- Models accel/gyro biases
- More realistic covariance propagation

State vector (16D):
    x = [
        px, py, pz,
        vx, vy, vz,
        qx, qy, qz, qw,
        bax, bay, baz,
        bgx, bgy, bgz
    ]

Observation:
    z = [px, py, pz]



    Cependant : 
    - Jacobienne encore simplifiée
    - Pas d'error-state EKF
    - Pas de propagation covariance attitude complète
    - Pas de modèle IMU discret exact
    - Pas de Mahalanobis gating
    - Pas de robustification numérique (Joseph form)
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_groundtruth(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def load_imu(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=[
            "idx", "timestamp",
            "gx", "gy", "gz",
            "ax", "ay", "az"
        ]
    )
    return df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Quaternion utilities
# ─────────────────────────────────────────────────────────────

def quat_normalize(q):
    return q / np.linalg.norm(q)


def integrate_quaternion(q, omega, dt):
    """
    Quaternion integration using small-angle approximation.
    omega: rad/s
    """
    angle = np.linalg.norm(omega) * dt

    if angle < 1e-12:
        return q

    axis = omega / np.linalg.norm(omega)

    dq = Rotation.from_rotvec(axis * angle).as_quat()

    q_new = (Rotation.from_quat(q) * Rotation.from_quat(dq)).as_quat()

    return quat_normalize(q_new)


# ─────────────────────────────────────────────────────────────
# EKF
# ─────────────────────────────────────────────────────────────

class DroneEKF:

    def __init__(self):

        self.n = 16

        self.x = np.zeros(self.n)

        # quaternion
        self.x[9] = 1.0

        self.P = np.eye(self.n) * 0.1

        # Process noise
        q_pos = 1e-4
        q_vel = 1e-2
        q_att = 1e-3
        q_ba  = 1e-5
        q_bg  = 1e-6

        self.Q = np.diag(
            [q_pos]*3 +
            [q_vel]*3 +
            [q_att]*4 +
            [q_ba]*3 +
            [q_bg]*3
        )

        # Position measurement noise
        self.R = np.diag([0.03, 0.03, 0.03])

        # Position observation model
        self.H = np.zeros((3, self.n))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        self.g = np.array([0, 0, -9.81])

    # ---------------------------------------------------------

    def predict(self, dt, accel_meas, gyro_meas):

        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        ba = self.x[10:13]
        bg = self.x[13:16]

        # Bias-corrected IMU
        accel = accel_meas - ba
        gyro  = gyro_meas  - bg

        # Update attitude
        q = integrate_quaternion(q, gyro, dt)

        # Rotate accel body -> world
        Rwb = Rotation.from_quat(q).as_matrix()

        a_world = Rwb @ accel + self.g

        # Kinematics
        p_new = p + v * dt + 0.5 * a_world * dt**2
        v_new = v + a_world * dt

        # Write back
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q

        # Linearized covariance propagation
        F = np.eye(self.n)

        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.P = F @ self.P @ F.T + self.Q

    # ---------------------------------------------------------

    def update(self, z):

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(self.n)

        self.P = (I - K @ self.H) @ self.P

        # normalize quaternion
        self.x[6:10] = quat_normalize(self.x[6:10])

        return y


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_ekf(gt_path, imu_path):

    gt = load_groundtruth(gt_path)
    imu = load_imu(imu_path)

    # Align time range
    t0 = max(gt.timestamp.iloc[0], imu.timestamp.iloc[0])
    t1 = min(gt.timestamp.iloc[-1], imu.timestamp.iloc[-1])

    gt = gt[(gt.timestamp >= t0) & (gt.timestamp <= t1)]
    imu = imu[(imu.timestamp >= t0) & (imu.timestamp <= t1)]

    # GT interpolation
    gt_interp = {
        ax: interp1d(
            gt.timestamp,
            gt[ax],
            bounds_error=False,
            fill_value="extrapolate"
        )
        for ax in ["x", "y", "z"]
    }

    ekf = DroneEKF()

    # Init position
    ekf.x[0] = gt.x.iloc[0]
    ekf.x[1] = gt.y.iloc[0]
    ekf.x[2] = gt.z.iloc[0]

    est = []

    t_prev = imu.timestamp.iloc[0]

    for i, row in imu.iterrows():

        t = row.timestamp
        dt = t - t_prev

        if dt <= 0:
            continue

        accel = np.array([row.ax, row.ay, row.az])
        gyro  = np.array([row.gx, row.gy, row.gz])

        # Prediction
        ekf.predict(dt, accel, gyro)

        # Update every 10 IMU steps
        if i % 10 == 0:

            z = np.array([
                gt_interp["x"](t),
                gt_interp["y"](t),
                gt_interp["z"](t)
            ])

            ekf.update(z)

        est.append(ekf.x[:3].copy())

        t_prev = t

    return np.array(est), gt


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_results(est, gt):

    gt_xyz = gt[["x", "y", "z"]].values

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gt_xyz[:, 0],
        gt_xyz[:, 1],
        gt_xyz[:, 2],
        label="Ground truth"
    )

    ax.plot(
        est[:, 0],
        est[:, 1],
        est[:, 2],
        label="EKF"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    plt.show()


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    GT_PATH  = "datasets/groundtruth.txt"
    IMU_PATH = "datasets/imu.txt"

    est, gt = run_ekf(GT_PATH, IMU_PATH)

    plot_results(est, gt)