import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class PlotDigitizer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not load image. Please check the path.")

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

        self.points = []
        self.axis_points = []
        self.data_points = []
        self.calibrated = False
        self.transform_matrix = None

        # Create buttons
        self.ax_clear = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.ax_calibrate = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.ax_export = plt.axes([0.59, 0.05, 0.1, 0.075])

        self.btn_clear = Button(self.ax_clear, 'Clear')
        self.btn_calibrate = Button(self.ax_calibrate, 'Calibrate')
        self.btn_export = Button(self.ax_export, 'Export Data')

        self.btn_clear.on_clicked(self.clear_points)
        self.btn_calibrate.on_clicked(self.calibrate_axes)
        self.btn_export.on_clicked(self.export_data)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.ax.set_title("Click to select points. First select axis points (x1,y1), (x2,y1), (x1,y2)")
        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if not self.calibrated:
            if len(self.axis_points) < 3:
                self.axis_points.append((x, y))
                self.ax.plot(x, y, 'ro' if len(self.axis_points) <= 1 else 'bo')
                self.ax.text(x, y, f"({x:.1f}, {y:.1f})", color='white')

                if len(self.axis_points) == 1:
                    self.ax.set_title("Now select a point on x-axis (x2,y1)")
                elif len(self.axis_points) == 2:
                    self.ax.set_title("Now select a point on y-axis (x1,y2)")
                elif len(self.axis_points) == 3:
                    self.ax.set_title("Click to select data points or press Calibrate")
            else:
                self.points.append((x, y))
                self.ax.plot(x, y, 'go')
                self.ax.text(x, y, f"({x:.1f}, {y:.1f})", color='white')
        else:
            self.points.append((x, y))
            plot_x, plot_y = self.transform_to_plot_coords(x, y)
            self.data_points.append((plot_x, plot_y))
            self.ax.plot(x, y, 'go')
            self.ax.text(x, y, f"({plot_x:.2f}, {plot_y:.2f})", color='white')

        self.fig.canvas.draw()

    def calibrate_axes(self, event=None):
        if len(self.axis_points) < 3:
            print("Please select all 3 axis points first")
            return

        # Get axis points
        (x1, y1), (x2, y2), (x3, y3) = self.axis_points[:3]

        # Ask user for actual plot coordinates
        plt.close(self.fig)
        plt.figure()

        x_min = float(input("Enter the x-coordinate for point (x1,y1): "))
        x_max = float(input("Enter the x-coordinate for point (x2,y1): "))
        y_min = float(input("Enter the y-coordinate for point (x1,y1): "))
        y_max = float(input("Enter the y-coordinate for point (x1,y2): "))

        # Calculate transformation matrix
        A = np.array([
            [x1, y1, 1, 0, 0, 0],
            [0, 0, 0, x1, y1, 1],
            [x2, y2, 1, 0, 0, 0],
            [0, 0, 0, x2, y2, 1],
            [x3, y3, 1, 0, 0, 0],
            [0, 0, 0, x3, y3, 1]
        ])

        b = np.array([x_min, y_min, x_max, y_min, x_min, y_max])

        try:
            # Solve for transformation parameters
            params = np.linalg.solve(A, b)
            self.transform_matrix = params.reshape(2, 3)
            self.calibrated = True

            # Transform existing points if any
            if self.points:
                self.data_points = [self.transform_to_plot_coords(x, y) for x, y in self.points]

            # Reopen the plot window
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(bottom=0.2)
            self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            # Redraw axis points
            for i, (x, y) in enumerate(self.axis_points):
                self.ax.plot(x, y, 'ro' if i == 0 else 'bo')
                self.ax.text(x, y, f"({x:.1f}, {y:.1f})", color='white')

            # Redraw data points with transformed coordinates
            for (img_x, img_y), (plot_x, plot_y) in zip(self.points, self.data_points):
                self.ax.plot(img_x, img_y, 'go')
                self.ax.text(img_x, img_y, f"({plot_x:.2f}, {plot_y:.2f})", color='white')

            # Recreate buttons
            self.ax_clear = plt.axes([0.7, 0.05, 0.1, 0.075])
            self.ax_calibrate = plt.axes([0.81, 0.05, 0.1, 0.075])
            self.ax_export = plt.axes([0.59, 0.05, 0.1, 0.075])

            self.btn_clear = Button(self.ax_clear, 'Clear')
            self.btn_calibrate = Button(self.ax_calibrate, 'Recalibrate')
            self.btn_export = Button(self.ax_export, 'Export Data')

            self.btn_clear.on_clicked(self.clear_points)
            self.btn_calibrate.on_clicked(self.calibrate_axes)
            self.btn_export.on_clicked(self.export_data)

            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

            self.ax.set_title("Digitizer calibrated. Click to add more points.")
            plt.show()

        except np.linalg.LinAlgError:
            print("Error in calibration. Please select proper axis points.")
            self.calibrated = False

    def transform_to_plot_coords(self, x, y):
        if not self.calibrated:
            return (x, y)

        vec = np.array([x, y, 1])
        transformed = self.transform_matrix @ vec
        return (transformed[0], transformed[1])

    def clear_points(self, event):
        self.points = []
        self.data_points = []
        self.axis_points = []
        self.calibrated = False
        self.transform_matrix = None

        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Click to select points. First select axis points (x1,y1), (x2,y1), (x1,y2)")
        self.fig.canvas.draw()

    def export_data(self, event):
        if not self.data_points:
            print("No data points to export")
            return

        filename = input("Enter filename to save (e.g., data.csv): ") or "data.csv"
        with open(filename, 'w') as f:
            f.write("x,y\n")
            for x, y in self.data_points:
                f.write(f"{x},{y}\n")
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_digitizer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    digitizer = PlotDigitizer(image_path)