import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Code to generate English diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

ax.add_patch(Rectangle((0.3, 0.8), 0.4, 0.1, fill=False))
ax.text(0.5, 0.85, 'MLP Policy Network', ha='center')

ax.add_patch(Rectangle((0.1, 0.4), 0.4, 0.1, fill=False))
ax.text(0.3, 0.45, 'Input: Graph Observation', ha='center')

ax.add_patch(Rectangle((0.5, 0.4), 0.4, 0.1, fill=False))
ax.text(0.7, 0.45, 'Output: Actions', ha='center')

ax.add_patch(Rectangle((0.25, 0.2), 0.5, 0.1, fill=False))
ax.text(0.5, 0.25, 'Training Params: n_steps=2048, lr=3e-4, gamma=0.99', ha='center')

ax.add_patch(FancyArrowPatch((0.3, 0.5), (0.5, 0.7), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.7, 0.5), (0.5, 0.7), arrowstyle='->'))
ax.add_patch(FancyArrowPatch((0.5, 0.8), (0.5, 0.3), arrowstyle='->'))

plt.title('PPO Agent Architecture Diagram')
plt.savefig('ppo_architecture.png')
plt.close()
# Explanation: Simple MLP diagram
