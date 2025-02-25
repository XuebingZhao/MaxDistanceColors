import matplotlib.pyplot as plt
import numpy as np

# Parameter settings
x = np.linspace(0, 10, 100)

# Manually set color list
# colors1 = ['#7c4522', '#41a5cf', '#781e73', '#145e4d', '#ca1727',
#            '#0d3881', '#65a733', '#da85a6', '#0a0406', '#e49d1c']  # CAM16LCD CMYK
# colors2 = ['#734c23', '#eade32', '#105577', '#da6f16', '#13572f',
#            '#d77a9c', '#479f37', '#390414', '#8f93bd', '#051011',
#            '#ada373', '#971670', '#259e9a', '#c81828', '#482674']

colors1 = ['#d35319', '#0e4a47', '#db8cad', '#eae047', '#0a0d25',
           '#4aaaa8', '#821f68', '#737d2f', '#1f72b0', '#4e2613']  # DIN99d
colors2 = ['#061626', '#af155d', '#eadf3d', '#635145', '#71b7d4',
           '#092c17', '#8476a9', '#92c19b', '#d1431b', '#472773',
           '#9d8427', '#146679', '#40050a', '#e49fa7', '#10733b']

# colors1 = ['#db7916', '#6e161b', '#4ea138', '#c51464', '#80bcdd',
#            '#183176', '#d5d227', '#6e6da5', '#0a0406', '#166034']   # Oklab
# colors2 = ['#7e1e73', '#128e6b', '#0a0406', '#a370a2', '#490714',
#            '#eab0b5', '#835626', '#ebe152', '#1a63a2', '#da7514',
#            '#222b6d', '#90b32b', '#0e4727', '#c91546', '#5dafd5']

# colors1 = ['#5a0e13', '#5ba437', '#d87ea1', '#e8ce27', '#0a2816',
#            '#d0401d', '#2b9f8f', '#562373', '#ac894f', '#1f7ab6']  # CIE Lab
# colors2 = ['#c51547', '#0a3c22', '#e5a719', '#0a0d22', '#f1e995',
#            '#292f7e', '#df8a7b', '#6ca933', '#a17dab', '#d45618',
#            '#33a4c8', '#5f1a1b', '#4ba782', '#941872', '#80732d']

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
