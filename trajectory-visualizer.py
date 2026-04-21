import matplotlib.pyplot as plt
import pandas as pd

DATA_GROUNDTRUTH = "datasets/groundtruth.txt"

# We will read the ground truth data from the file and store it in a DataFrame
# The file is composed of 8 columns: timestamp, x, y, z, qx, qy, qz, qw
# We will first simply show the trajectory in 3D space using the x, y, z columns

def read_groundtruth(file_path):
    # Read the data from the file using pandas
    df = pd.read_csv(file_path, sep=" ", header=None, comment="#",
                     names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
    return df

def plot_trajectory(df):
    # Create a 3D plot of the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory using x, y, z columns
    ax.plot(df['x'], df['y'], df['z'], label='Trajectory', marker='o', markersize=2, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Visualization')
    ax.legend()

    # Show the plot
    plt.show()

def groundtruth_extented(filepath):
    """
    Extend the ground truth data with velocity and acceleration information
    """
    df = read_groundtruth(filepath)
    df["vx"] = df["x"].diff()
    df["vy"] = df["y"].diff()
    df["vz"] = df["z"].diff()
    df["ax"] = df["vx"].diff()
    df["ay"] = df["vy"].diff()
    df["az"] = df["vz"].diff()
    return df

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_w_velocity(df, step=100):
    """
    Plot the ground truth data with velocity information.
    The color of the trajectory reflects the velocity magnitude.
    
    Parameters:
    - df: DataFrame with columns x, y, z, vx, vy, vz
    - step: subsampling factor for quiver (to avoid clutter)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Compute velocity magnitude
    v = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

    # Plot trajectory with color mapping
    scatter = ax.scatter(df['x'], df['y'], df['z'], c=v, cmap='viridis', s=5)

    # Subsample for quiver
    idx = np.arange(0, len(df), step)

    ax.quiver(
        df['x'].iloc[idx],
        df['y'].iloc[idx],
        df['z'].iloc[idx],
        df['vx'].iloc[idx],
        df['vy'].iloc[idx],
        df['vz'].iloc[idx],
        length=0.1,
        normalize=True
    )

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Velocity magnitude')

    plt.show()

def plot_trajectory_w_acceleration(df, step=100):
    """
    Plot the ground truth data with acceleration information.
    The color of the trajectory reflects the acceleration magnitude.
    
    Parameters:
    - df: DataFrame with columns x, y, z, ax, ay, az
    - step: subsampling factor for quiver (to avoid clutter)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Compute acceleration magnitude
    a = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

    # Plot trajectory with color mapping
    scatter = ax.scatter(df['x'], df['y'], df['z'], c=a, cmap='viridis', s=5)

    # Subsample for quiver
    idx = np.arange(0, len(df), step)

    ax.quiver(
        df['x'].iloc[idx],
        df['y'].iloc[idx],
        df['z'].iloc[idx],
        df['ax'].iloc[idx],
        df['ay'].iloc[idx],
        df['az'].iloc[idx],
        length=0.1,
        normalize=True
    )

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Acceleration magnitude')

    plt.show()

if __name__ == "__main__":
    df = groundtruth_extented(DATA_GROUNDTRUTH)
    plot_trajectory_w_acceleration(df)
    plot_trajectory_w_velocity(df)