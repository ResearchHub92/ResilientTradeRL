import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
plt.plot(df['Year'], df['ΔSR_Shock'], marker='o', label='SR Drift After Shock')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Year')
plt.ylabel('SR Drift')
plt.title('SR Drift Over Years')
plt.legend()
plt.savefig('sr_drift_line.png')
plt.close()