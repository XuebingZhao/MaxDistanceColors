import matplotlib.pyplot as plt
import numpy as np

# Parameter settings
x = np.linspace(0, 10, 100)

# Manually set color list
# colors1 = ['#2eabfe', '#04b29a', '#fe992c', '#aefe06', '#fe0569',
#            '#555800', '#084a5b', '#8e00fd', '#550001', '#00004c']   # DIN99d
# colors1 = ['#00b27e', '#f0fe01', '#04a5fd', '#be7900', '#fe03d2',
#            '#6e0afe', '#fe0202', '#013300', '#590005', '#00003b']   # CAM16LCD

colors1 = ['#d0401b', '#033029', '#991670', '#846227', '#352b7d',
           '#eade35', '#239ca9', '#370509', '#3f9d39', '#a486b1']   # CAM16LCD CMYK
colors2 = ['#0e6a79', '#de8b98', '#153379', '#aac024', '#692173',
           '#1b9151', '#6d191b', '#75b9d6', '#092e18', '#d1461a',
           '#7f75a8', '#e29418', '#190718', '#bf126e', '#7a653e']


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
