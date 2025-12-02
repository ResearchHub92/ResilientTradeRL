import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Code to generate English schema
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Boxes
ax.add_patch(Rectangle((0.1, 0.7), 0.3, 0.1, fill=False))
ax.text(0.25, 0.75, 'Load Filtered Matrix', ha='center', va='center')

ax.add_patch(Rectangle((0.5, 0.7), 0.3, 0.1, fill=False))
ax.text(0.65, 0.75, 'Compute Spearman Correlation', ha='center', va='center')

ax.add_patch(Rectangle((0.1, 0.4), 0.3, 0.1, fill=False))
ax.text(0.25, 0.45, 'Rank In/Out Strengths', ha='center', va='center')

ax.add_patch(Rectangle((0.5, 0.4), 0.3, 0.1, fill=False))
ax.text(0.65, 0.45, 'Calculate SR (Weighted Rank)', ha='center', va='center')

ax.add_patch(Rectangle((0.3, 0.1), 0.4, 0.1, fill=False))
ax.text(0.5, 0.15, 'Extract Top Complement Pairs', ha='center', va='center')

# Arrows
ax.add_patch(FancyArrowPatch((0.4, 0.75), (0.5, 0.75), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.25, 0.7), (0.25, 0.5), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.65, 0.7), (0.65, 0.5), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.25, 0.4), (0.5, 0.2), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.65, 0.4), (0.5, 0.2), arrowstyle='->'))

plt.title('SR Calculation and Complement Pairs Schema')
plt.savefig('sr_calc_schema.png')
plt.close()
# Explanation: This code generates a simple schema. If you want to be more realistic, use 2014 data in create_sector_complements.py.
