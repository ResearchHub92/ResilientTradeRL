import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import sem, t

df = pd.read_csv('results/shock_results_filtered.csv')
mean = df['After_SR'].mean()
ci = sem(df['After_SR']) * t.ppf((1 + 0.95) / 2, len(df)-1)
plt.errorbar(df['Year'], df['After_SR'], yerr=ci, fmt='o-')
plt.xlabel('Year')
plt.ylabel('After SR with 95% CI')
plt.title('Confidence Interval Line Plot')
plt.savefig('ci_line.png')
plt.close()