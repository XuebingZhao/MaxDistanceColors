import matplotlib.pyplot as plt
import numpy as np

# Parameter settings
x = np.linspace(0, 10, 100)

# Manually set color list
# colors1 = ['#2eabfe', '#04b29a', '#fe992c', '#aefe06', '#fe0569',
#            '#555800', '#084a5b', '#8e00fd', '#550001', '#00004c']   # DIN99d
# colors1 = ['#00b27e', '#f0fe01', '#04a5fd', '#be7900', '#fe03d2',
#            '#6e0afe', '#fe0202', '#013300', '#590005', '#00003b']   # CAM16LCD

colors1 = ['#a385b3', '#ecbe2b', '#259da9', '#49a037', '#7f5e26',
           '#d7461f', '#9a1570', '#3a2778', '#033029', '#38050a']   # CAM16LCD CMYK
colors2 = ['#74b7d7', '#ed98a4', '#ea9e1b', '#8776a9', '#9cbc2d',
           '#0f6877', '#7f5925', '#1c7b3a', '#d34e18', '#c5176c',
           '#6c2170', '#1a3076', '#132f17', '#6e1519', '#1d0815']


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
