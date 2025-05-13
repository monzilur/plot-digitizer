import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_and_plot(csv_file):
    # Load the CSV file
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check if the required columns exist
    if not all(col in data.columns for col in ['x', 'y']):
        print("Error: CSV file must contain 'x' and 'y' columns.")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['x'], data['y'], 'bo-', label='Digitized Data')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Digitized Data Plot')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Optionally print the data
    print("\nLoaded Data:")
    print(data)


if __name__ == "__main__":
    import sys

    # Get filename from command line or use default
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "data.csv"  # default filename

    load_and_plot(csv_file)
