import matplotlib.pyplot as plt
import numpy as np

# Parameter settings
n_line = 10
x = np.linspace(0, 10, 100)
y = [np.sin(x + 2 * np.pi * i / n_line) for i in range(n_line)]

# Manually set color list
colors = ['#2eabfe', '#04b29a', '#fe992c', '#aefe06', '#fe0569', '#555800', '#084a5b', '#8e00fd', '#550001', '#00004c']   # DIN99d
# colors = ['#de9e02', '#ff86e7', '#13aec8', '#5afe07', '#5c53fe', '#0d8403', '#ec0104', '#520005', '#001e00', '#01005d']   # CAM16UCS

# Create figure and subplots
plt.figure(figsize=(12, 4))  # Set figure size

# Left subplot: default color sequence
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
for i in range(n_line):
    plt.plot(x, y[i], label=f'Line {i + 1}', lw=2)
plt.title('Default Colors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Right subplot: manually set color list
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
for i in range(n_line):
    plt.plot(x, y[i], color=colors[i], label=f'Line {i + 1}', lw=2)
plt.title('Max Distance Colors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Display the figure
plt.tight_layout()  # Automatically adjust subplot spacing
plt.show()
