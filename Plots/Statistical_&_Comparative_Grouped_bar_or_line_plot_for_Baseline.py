import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')  # With Baseline_SR
plt.bar(df['Year'] - 0.2, df['After_SR'], width=0.4, label='PPO')
plt.bar(df['Year'] + 0.2, df['Baseline_SR'], width=0.4, label='Baseline')
plt.xlabel('Year')
plt.ylabel('SR')
plt.title('Baseline Comparison Grouped Bar')
plt.legend()
plt.savefig('baseline_comparison_bar.png')
plt.close()
