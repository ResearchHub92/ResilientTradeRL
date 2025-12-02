import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Code to generate English schema
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

ax.add_patch(Rectangle((0.08, 0.7), 0.33, 0.1, fill=False))
ax.text(0.25, 0.75, 'Observation Space (Graph State)', ha='center')

ax.add_patch(Rectangle((0.48, 0.7), 0.33, 0.1, fill=False))
ax.text(0.65, 0.75, 'Action Space (Remove/Add Edges)', ha='center')

ax.add_patch(Rectangle((0.3, 0.4), 0.4, 0.1, fill=False))
ax.text(0.5, 0.45, 'Reset: Load Year & Apply Shock', ha='center')

ax.add_patch(Rectangle((0.3, 0.1), 0.4, 0.1, fill=False))
ax.text(0.5, 0.15, 'Step: Apply Action & Compute Reward', ha='center')

ax.add_patch(FancyArrowPatch((0.25, 0.7), (0.5, 0.5), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.65, 0.7), (0.5, 0.5), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.5, 0.4), (0.5, 0.2), arrowstyle='->'))

plt.title('Gym Environment Structure Schema')
plt.savefig('gym_schema.png')
plt.close()
# Explanation: Simple schema.
