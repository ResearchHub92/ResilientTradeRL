import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
years = df['Year']
before_sr = df['Before_SR']
shock_sr = df['Shock_SR']
after_sr = df['After_SR']
before_pearson = df['Before_Pearson']
shock_pearson = df['Shock_Pearson']
after_pearson = df['After_Pearson']

plt.figure(figsize=(12, 8))

# SR chart
plt.subplot(2, 1, 1)
plt.plot(years, before_sr, 'g-', marker='o', linewidth=2, label='Before Shock')
plt.plot(years, shock_sr, 'r-', marker='s', linewidth=2, label='During Shock')
plt.plot(years, after_sr, 'b-', marker='^',
         linewidth=2, label='After Recovery')
plt.ylabel('Stability Ratio (SR)')
plt.title('Network Stability Evolution - Before, During and After Shock')
plt.legend()
plt.grid(True, alpha=0.3)

# Pearson chart
plt.subplot(2, 1, 2)
plt.plot(years, before_pearson, 'g-', marker='o',
         linewidth=2, label='Before Shock')
plt.plot(years, shock_pearson, 'r-', marker='s',
         linewidth=2, label='During Shock')
plt.plot(years, after_pearson, 'b-', marker='^',
         linewidth=2, label='After Recovery')
plt.ylabel('Pearson Correlation')
plt.xlabel('Year')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shock_evolution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
