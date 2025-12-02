import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Code to generate English diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

ax.add_patch(Rectangle((0.2, 0.8), 0.6, 0.1, fill=False))
ax.text(0.5, 0.85, 'Compute Distance: |Current SR - Target SR|', ha='center')

ax.add_patch(Rectangle((0.1, 0.5), 0.4, 0.1, fill=False))
ax.text(0.3, 0.55, 'SR Drift Penalty', ha='center')

ax.add_patch(Rectangle((0.5, 0.5), 0.4, 0.1, fill=False))
ax.text(0.7, 0.55, 'Latency Penalty', ha='center')

ax.add_patch(Rectangle((0.3, 0.2), 0.4, 0.1, fill=False))
ax.text(0.5, 0.25, 'Collapse Penalty', ha='center')

ax.add_patch(FancyArrowPatch((0.5, 0.8), (0.3, 0.6), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.5, 0.8), (0.7, 0.6), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.3, 0.5), (0.5, 0.3), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.7, 0.5), (0.5, 0.3), arrowstyle='->'))

plt.title('Reward Function Diagram')
plt.savefig('reward_function_diagram.png')
plt.close()
# Explanation: Reward composition diagram.
