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

if __name__ == "__main__":
    plot_trajectory(read_groundtruth(DATA_GROUNDTRUTH))