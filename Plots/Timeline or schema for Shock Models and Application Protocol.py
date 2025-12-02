import matplotlib.pyplot as plt

# Code to generate English timeline
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Timeline
ax.plot([0, 10], [0.5, 0.5], 'k-')

# Points
ax.plot(2, 0.5, 'o')
ax.text(2, 0.6, 'Reset Environment', ha='center')

ax.plot(4, 0.5, 'o')
ax.text(4, 0.6, 'Apply Shock (Random 20-40% Edge Removal)', ha='center')

ax.plot(6, 0.5, 'o')
ax.text(6, 0.6, 'Targeted Shocks (US-MEX, CHN-TWN)', ha='center')

ax.plot(8, 0.5, 'o')
ax.text(8, 0.6, 'Mutation Shocks (Random Mutations)', ha='center')

plt.title('Shock Application Timeline')
plt.savefig('shock_timeline.png')
plt.close()
# Explanation: This timeline is simple. The current code only has one type of shock; for more types, expand _apply_shock.
