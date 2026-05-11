"""
Extended Kalman Filter for drone trajectory prediction.
Fuses IMU data (angular velocity + linear acceleration) with ground truth positions.

State vector: [x, y, z, vx, vy, vz, ax, ay, az]  (9D)
Observations:  [x, y, z]                           (3D, from groundtruth)

Usage:
    python ekf_drone.py --gt datasets/groundtruth.txt --imu datasets/unimu.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_groundtruth(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#',
                     names=['timestamp','x','y','z','qx','qy','qz','qw'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_imu(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#',
                     names=['idx','timestamp',
                            'ang_vel_x','ang_vel_y','ang_vel_z',
                            'lin_acc_x','lin_acc_y','lin_acc_z'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# EKF — state: [x, y, z, vx, vy, vz, ax, ay, az]
# ─────────────────────────────────────────────────────────────────────────────

class DroneEKF:
    """
    Extended Kalman Filter for drone dead-reckoning + position correction.

    Prediction step  → physics model using IMU accelerations
    Update step      → position correction from ground truth (or GPS)
    """

    def __init__(self, x0, P0, Q, R):
        """
        Args:
            x0  : initial state (9,)     [x,y,z, vx,vy,vz, ax,ay,az]
            P0  : initial covariance (9,9)
            Q   : process noise (9,9)    — how much we trust the model
            R   : measurement noise (3,3)— how much we trust position obs
        """
        self.x = x0.copy()         # state estimate
        self.P = P0.copy()         # state covariance
        self.Q = Q                 # process noise covariance
        self.R = R                 # measurement noise covariance

        # Observation matrix: we only observe position (x, y, z)
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def _state_transition(self, dt, imu_acc):
        """
        Physics model: constant-acceleration kinematics.
        The IMU acceleration is used to update the acceleration state,
        then propagated through to velocity and position.

        x_{k+1} = F x_k  (with acceleration input from IMU)
        """
        F = np.eye(9)
        # position ← position + velocity*dt + 0.5*acc*dt²
        F[0, 3] = dt;  F[0, 6] = 0.5 * dt**2
        F[1, 4] = dt;  F[1, 7] = 0.5 * dt**2
        F[2, 5] = dt;  F[2, 8] = 0.5 * dt**2
        # velocity ← velocity + acceleration*dt
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt
        # acceleration ← stays (updated by IMU below)

        x_pred = F @ self.x

        # Override acceleration with fresh IMU reading
        x_pred[6] = imu_acc[0]
        x_pred[7] = imu_acc[1]
        x_pred[8] = imu_acc[2]

        return x_pred, F

    def predict(self, dt, imu_acc):
        """Prediction step: propagate state with IMU acceleration."""
        x_pred, F = self._state_transition(dt, imu_acc)
        P_pred = F @ self.P @ F.T + self.Q

        self.x = x_pred
        self.P = P_pred

    def update(self, z):
        """
        Update step: correct with position measurement z = [x, y, z].

        Innovation:    y = z - H x_pred
        Kalman gain:   K = P H^T (H P H^T + R)^{-1}
        State update:  x = x_pred + K y
        Cov update:    P = (I - K H) P
        """
        y = z - self.H @ self.x                             # innovation
        S = self.H @ self.P @ self.H.T + self.R             # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)            # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P

        return y, K                                          # useful for diagnostics

    def predict_ahead(self, n_steps, dt):
        """
        Pure prediction (no update) — used for 'real-time' forward horizon.
        Returns predicted positions over the next n_steps.
        """
        x = self.x.copy()
        P = self.P.copy()
        F = np.eye(9)
        F[0,3]=dt; F[0,6]=0.5*dt**2
        F[1,4]=dt; F[1,7]=0.5*dt**2
        F[2,5]=dt; F[2,8]=0.5*dt**2
        F[3,6]=dt; F[4,7]=dt; F[5,8]=dt

        positions = []
        for _ in range(n_steps):
            x = F @ x
            positions.append(x[:3].copy())
        return np.array(positions)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ekf(gt_path, imu_path,
            imu_update_ratio=10,     # use 1 GT position every N IMU steps
            prediction_horizon=50,   # steps to predict ahead
            use_gt_updates=True):    # False = pure dead-reckoning (no GT)
    """
    Main EKF loop.

    The IMU runs at ~1 kHz; ground truth at ~100 Hz.
    We interpolate GT timestamps to align with IMU.
    """

    print("Loading data...")
    gt  = load_groundtruth(gt_path)
    imu = load_imu(imu_path)

    # ── Align time ranges ────────────────────────────────────────────────────
    t_start = max(gt['timestamp'].iloc[0],  imu['timestamp'].iloc[0])
    t_end   = min(gt['timestamp'].iloc[-1], imu['timestamp'].iloc[-1])

    gt  = gt[(gt['timestamp']  >= t_start) & (gt['timestamp']  <= t_end)].reset_index(drop=True)
    imu = imu[(imu['timestamp'] >= t_start) & (imu['timestamp'] <= t_end)].reset_index(drop=True)

    if len(imu) == 0 or len(gt) == 0:
        raise ValueError("No overlapping timestamps between IMU and ground truth.")

    print(f"  IMU samples : {len(imu)}")
    print(f"  GT  samples : {len(gt)}")

    # ── Initialisation ────────────────────────────────────────────────────────
    x0 = np.zeros(9)
    x0[0] = gt['x'].iloc[0]
    x0[1] = gt['y'].iloc[0]
    x0[2] = gt['z'].iloc[0]

    # Process noise — tune these to your drone dynamics
    q_pos  = 0.01   # position drift
    q_vel  = 0.1    # velocity drift
    q_acc  = 1.0    # acceleration drift (IMU noise dominates)

    Q = np.diag([q_pos]*3 + [q_vel]*3 + [q_acc]*3)

    # Measurement noise — how noisy is your position sensor?
    r_pos = 0.05    # 5 cm position noise (UZH dataset is quite precise)
    R = np.diag([r_pos]*3)

    P0 = np.diag([0.1]*3 + [1.0]*3 + [5.0]*3)

    ekf = DroneEKF(x0, P0, Q, R)

    # ── GT interpolator (for update step) ────────────────────────────────────
    from scipy.interpolate import interp1d
    gt_interp = {
        ax: interp1d(gt['timestamp'], gt[ax], bounds_error=False, fill_value='extrapolate')
        for ax in ['x','y','z']
    }

    # ── Main loop ─────────────────────────────────────────────────────────────
    estimated   = []   # filtered trajectory
    predicted   = []   # pure prediction (no GT)
    innovations = []   # innovation norm (filter health)
    timestamps  = []

    t_prev = imu['timestamp'].iloc[0]

    print("Running EKF...")
    for i, row in imu.iterrows():
        t     = row['timestamp']
        dt    = t - t_prev
        if dt <= 0:
            continue

        imu_acc = np.array([row['lin_acc_x'], row['lin_acc_y'], row['lin_acc_z']])

        # Gravity removal (approximation — proper version needs orientation)
        # The IMU measures specific force = acceleration - gravity
        # For now we subtract gravity along the dominant axis (z≈-9.8)
        GRAVITY = np.array([0.0, 0.0, -9.81])
        world_acc = imu_acc + GRAVITY      # crude: works when drone is mostly flat

        # ── Predict ──────────────────────────────────────────────────────────
        ekf.predict(dt, world_acc)

        # ── Update (every imu_update_ratio steps) ────────────────────────────
        if use_gt_updates and (i % imu_update_ratio == 0):
            z = np.array([gt_interp[ax](t) for ax in ['x','y','z']])
            inno, K = ekf.update(z)
            innovations.append(np.linalg.norm(inno))

        estimated.append(ekf.x[:3].copy())
        timestamps.append(t)
        t_prev = t

        if i % 5000 == 0:
            print(f"  Step {i}/{len(imu)}", end='\r')

    print(f"\nDone. {len(estimated)} steps processed.")

    estimated   = np.array(estimated)
    gt_xyz      = gt[['x','y','z']].values
    timestamps  = np.array(timestamps)

    # ── Forward prediction from last state ───────────────────────────────────
    dt_mean = np.mean(np.diff(timestamps[-100:]))
    future_positions = ekf.predict_ahead(prediction_horizon, dt_mean)

    return {
        'estimated'       : estimated,
        'groundtruth'     : gt_xyz,
        'future'          : future_positions,
        'timestamps'      : timestamps,
        'gt_timestamps'   : gt['timestamp'].values,
        'innovations'     : np.array(innovations) if innovations else np.array([]),
        'final_state'     : ekf.x,
        'final_cov_diag'  : np.diag(ekf.P),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results):
    est = results['estimated']
    gt  = results['groundtruth']
    fut = results['future']
    inn = results['innovations']

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('EKF — Drone trajectory estimation & prediction', fontsize=14)

    # ── 3D trajectory ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
    ax1.plot(gt[:,0],  gt[:,1],  gt[:,2],  'k-',  lw=0.8, alpha=0.5, label='Ground truth')
    ax1.plot(est[:,0], est[:,1], est[:,2], 'b-',  lw=1.0, alpha=0.8, label='EKF estimate')
    ax1.plot(fut[:,0], fut[:,1], fut[:,2], 'r--', lw=1.5, label=f'Prediction ({len(fut)} steps)')
    ax1.scatter(*est[-1],  c='blue',  s=30, zorder=5)
    ax1.scatter(*fut[-1],  c='red',   s=50, zorder=5, label='Predicted end')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.legend(fontsize=8)
    ax1.set_title('3D trajectory')

    # ── Per-axis comparison ───────────────────────────────────────────────────
    t_est = results['timestamps']
    t_gt  = results['gt_timestamps']
    # Align to relative time
    t0    = t_est[0]
    t_rel = t_est - t0
    tg_rel = t_gt - t0

    for idx, (axis, col) in enumerate([('X','b'),('Y','g'),('Z','r')]):
        ax = fig.add_subplot(2, 3, idx + 2)   # top row: 2,3,4 (skip 1 taken by 3D)
        # Avoid 2,3 overlap: use right column
        ax = fig.add_subplot(2, 3, idx + 2)

        gt_col = gt[:, idx]
        ax.plot(tg_rel[:len(gt_col)], gt_col,  'k-',  lw=0.7, alpha=0.5, label='GT')
        ax.plot(t_rel, est[:, idx],  c=col, lw=1.0, alpha=0.9, label='EKF')
        ax.set_title(f'{axis} position (m)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Innovation norm ───────────────────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 6)
    if len(inn) > 0:
        ax5.plot(inn, 'orange', lw=1.0)
        ax5.set_title('Innovation norm (filter health)')
        ax5.set_xlabel('Update step')
        ax5.set_ylabel('||y|| (m)')
        ax5.axhline(np.mean(inn), color='red', ls='--', lw=0.8, label=f'Mean: {np.mean(inn):.3f} m')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No updates\n(pure dead-reckoning)',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Innovation norm')

    plt.tight_layout()
    plt.savefig('ekf_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Print summary ─────────────────────────────────────────────────────────
    from scipy.interpolate import interp1d
    t_gt  = results['gt_timestamps']
    t0    = results['timestamps'][0]
    gt_interp_x = interp1d(t_gt, gt[:,0], bounds_error=False, fill_value='extrapolate')
    gt_interp_y = interp1d(t_gt, gt[:,1], bounds_error=False, fill_value='extrapolate')
    gt_interp_z = interp1d(t_gt, gt[:,2], bounds_error=False, fill_value='extrapolate')

    t_eval = results['timestamps'][::10]
    gt_at_t = np.column_stack([
        gt_interp_x(t_eval),
        gt_interp_y(t_eval),
        gt_interp_z(t_eval),
    ])
    est_at_t = results['estimated'][::10]
    n = min(len(gt_at_t), len(est_at_t))
    rmse = np.sqrt(np.mean(np.sum((gt_at_t[:n] - est_at_t[:n])**2, axis=1)))

    print("\n── EKF Summary ──────────────────────────────────")
    print(f"  RMSE (position)      : {rmse:.4f} m")
    print(f"  Final state x,y,z    : {results['final_state'][:3]}")
    print(f"  Final state vx,vy,vz : {results['final_state'][3:6]}")
    print(f"  State covariance diag: {results['final_cov_diag'][:6].round(4)}")
    if len(inn) > 0:
        print(f"  Mean innovation      : {np.mean(inn):.4f} m")
    print(f"  Predicted next pos   : {results['future'][-1].round(3)} m")
    print("─────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EKF drone trajectory estimation')
    parser.add_argument('--gt',      default='datasets/groundtruth.txt',
                        help='Path to groundtruth.txt')
    parser.add_argument('--imu',     default='datasets/unimu.txt',
                        help='Path to unimu.txt')
    parser.add_argument('--horizon', type=int, default=50,
                        help='Number of future steps to predict')
    parser.add_argument('--no-updates', action='store_true',
                        help='Disable GT updates (pure IMU dead-reckoning)')    
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        print(f"Ground truth not found: {args.gt}")
        exit(1)
    if not os.path.exists(args.imu):
        print(f"IMU file not found: {args.imu}")
        exit(1)

    results = run_ekf(
        gt_path=args.gt,
        imu_path=args.imu,
        prediction_horizon=args.horizon,
        use_gt_updates=not args.no_updates,
    )
    plot_results(results)
# python ekf_drone.py --gt datasets/groundtruth.txt --imu datasets/imu.txt --horizon 50