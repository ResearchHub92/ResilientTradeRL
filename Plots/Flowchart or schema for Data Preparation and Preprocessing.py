import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Code to generate English flowchart
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

# Boxes
ax.add_patch(Rectangle((0.3, 0.8), 0.43, 0.1, fill=False))
ax.text(0.515, 0.85, 'Load WIOD Data', ha='center', va='center')

ax.add_patch(Rectangle((0.3, 0.6), 0.43, 0.1, fill=False))
ax.text(0.515, 0.65, 'Filter Countries & Sectors', ha='center', va='center')

ax.add_patch(Rectangle((0.3, 0.4), 0.43, 0.1, fill=False))
ax.text(0.515, 0.45, 'Remove Zeros & Final Demand', ha='center', va='center')

ax.add_patch(Rectangle((0.3, 0.2), 0.43, 0.1, fill=False))
ax.text(0.515, 0.25, 'Normalize & Label Nodes', ha='center', va='center')

ax.add_patch(Rectangle((0.3, 0.0), 0.43, 0.1, fill=False))
ax.text(0.515, 0.05, 'Convert to Weighted Directed Graph',
        ha='center', va='center')

# Arrows
ax.add_patch(FancyArrowPatch((0.5, 0.8), (0.5, 0.7), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.5, 0.6), (0.5, 0.5), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.5, 0.4), (0.5, 0.3), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.5, 0.2), (0.5, 0.1), arrowstyle='->'))

plt.title('Data Preparation Flowchart')
plt.savefig('data_preprocessing_flowchart.png')
plt.close()
# Explanation: This code generates and saves a simple flowchart. Run it to take a screenshot.
