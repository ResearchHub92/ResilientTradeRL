import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

# Code to generate English diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Boxes
ax.add_patch(Rectangle((0.1, 0.7), 0.2, 0.1, fill=False))
ax.text(0.2, 0.75, 'Data Module', ha='center')

ax.add_patch(Rectangle((0.4, 0.7), 0.2, 0.1, fill=False))
ax.text(0.5, 0.75, 'Environment Module', ha='center')

ax.add_patch(Rectangle((0.7, 0.7), 0.2, 0.1, fill=False))
ax.text(0.8, 0.75, 'Agent Module', ha='center')

ax.add_patch(Rectangle((0.4, 0.4), 0.2, 0.1, fill=False))
ax.text(0.5, 0.45, 'Evaluation Module', ha='center')

# Connections
ax.add_patch(ConnectionPatch((0.2, 0.7), (0.5, 0.5),
             'data', 'data', arrowstyle='->'))
ax.add_patch(ConnectionPatch((0.5, 0.7), (0.5, 0.5),
             'data', 'data', arrowstyle='->'))
ax.add_patch(ConnectionPatch((0.8, 0.7), (0.5, 0.5),
             'data', 'data', arrowstyle='->'))

plt.title('Modular Project Architecture Diagram')
plt.savefig('modular_architecture.png')
plt.close()
# Explanation: Project structure diagram.
