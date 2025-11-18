# Import required libraries
import numpy as np                  # For numeric operations
import csv                          # To write CSV files
from SimulationEnvironment import SimulationEnvironment  # Robot simulator
import time                         # To add delays if needed

def collect_data(num_samples=20000):
    """
    Collects robot sensor readings and collision labels from the simulation.
    Saves them to training_data.csv.
    """

    # Create the simulation environment
    sim = SimulationEnvironment()

    # Open a new CSV file for writing the dataset
    with open("saved/training_data.csv", "w", newline="") as file:

        # Create a CSV writer object
        writer = csv.writer(file)

        # Loop to collect the requested number of samples
        for i in range(num_samples):

            # Reset robot to a new random position
            sim.reset()

            # Run one simulation step and get sensor readings
            features = sim.get_sensor_readings()

            # Ask the simulator whether this state caused a collision
            collision = sim.did_collide()

            # Combine sensor readings and collision label into one row
            row = list(features) + [int(collision)]

            # Write the row to the CSV file
            writer.writerow(row)

            # Print progress every 1000 samples (optional)
            if i % 1000 == 0:
                print(f"Collected {i}/{num_samples} samples...")

    print("Data collection complete!")

# If file is run from terminal, start collecting data
if __name__ == "__main__":
    collect_data(20000)
