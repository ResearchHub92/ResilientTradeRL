import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
plt.hist(df['Recovery_Steps'], bins=5)
plt.xlabel('Recovery Steps')
plt.ylabel('Frequency')
plt.title('Histogram of Recovery Steps')
plt.savefig('recovery_steps_hist.png')
plt.close()