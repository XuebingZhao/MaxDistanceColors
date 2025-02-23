import matplotlib.pyplot as plt
import numpy as np

# Parameter settings
x = np.linspace(0, 10, 100)

# Manually set color list
# colors1 = ['#2eabfe', '#04b29a', '#fe992c', '#aefe06', '#fe0569',
#            '#555800', '#084a5b', '#8e00fd', '#550001', '#00004c']   # DIN99d
# colors1 = ['#00b27e', '#f0fe01', '#04a5fd', '#be7900', '#fe03d2',
#            '#6e0afe', '#fe0202', '#013300', '#590005', '#00003b']   # CAM16LCD

colors1 = ['#7c4522', '#41a5cf', '#781e73', '#145e4d', '#ca1727',
           '#0d3881', '#65a733', '#da85a6', '#0a0406', '#e49d1c']  # CAM16LCD CMYK
colors2 = ['#734c23', '#eade32', '#105577', '#da6f16', '#13572f',
           '#d77a9c', '#479f37', '#390414', '#8f93bd', '#051011',
           '#ada373', '#971670', '#259e9a', '#c81828', '#482674']

# Create figure and subplots
plt.figure(figsize=(12, 4.5))  # Set figure size

# Left subplot
plt.subplot(1, 2, 1)
n_line = 10
y = [np.sin(x + 2 * np.pi * i / n_line) for i in range(n_line)]
for i in range(n_line):
    plt.plot(x, y[i], color=colors1[i], label=f'Line {i + 1}', lw=2)
plt.title(f'{n_line} Max Distance Colors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Right subplot
plt.subplot(1, 2, 2)
n_line = 15
y = [np.sin(x + 2 * np.pi * i / n_line) for i in range(n_line)]
for i in range(n_line):
    plt.plot(x, y[i], color=colors2[i], label=f'Line {i + 1}', lw=2)
plt.title(f'{n_line} Max Distance Colors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Display the figure
plt.tight_layout()  # Automatically adjust subplot spacing
plt.show()
